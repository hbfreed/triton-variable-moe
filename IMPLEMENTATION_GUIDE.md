# Triton MoE Implementation Guide

A roadmap for implementing a high-performance MoE layer in pure Triton, with support for variable-sized experts.

## Overview

The goal: replace `stk.ops.sdd/dsd` and megablocks ops with custom Triton kernels that use the **grouped dense GEMM** pattern (direct indexing) while supporting **variable-sized experts**.

---

## Why Direct Indexing? (Motivation)

The megablocks/stk approach uses block-sparse matrices to represent "which tokens go to which experts." This requires building sparse matrix metadata every forward pass:

**Profiling the reference implementation shows:**
```
topo_setup         ~5%   - compute padded bins
topo_column_idx   ~20%   - build column indices (topology_var kernel + repeat_interleave)
topo_row_idx       ~3%   - build row indices
topo_transpose     ~8%   - transpose sparse structure for backward
topo_matrix        ~2%   - package into stk.Matrix
─────────────────────────
TOTAL TOPOLOGY    ~38%   of forward pass time!
```

**The direct indexing insight:** All of this metadata just encodes "token i uses weight columns [offset_j, offset_j + width_j) for expert j." With Triton, we can compute this on-the-fly:

```python
# Sparse matrix approach: build indices, then sparse GEMM uses them
column_indices = topology_var(...)  # expensive!
output = stk.ops.sdd(x, W, topology)

# Direct indexing approach: just pass offsets, kernel figures it out
# expert_offsets = [0, 512, 1024, 1536, ...]  (precomputed at init)
# expert_widths = [512, 512, 256, ...]         (precomputed at init)
output = grouped_gemm(x, W, expert_offsets, expert_widths, tokens_per_expert)
```

The kernel directly indexes into the concatenated weight matrix—no sparse structure needed.

### Memory Coalescing (Why This Doesn't Hurt Performance)

You might worry: "Won't random expert assignments cause scattered memory access?"

No—because **sorting is what gives you coalescing**, not the sparse matrix:

```
BEFORE sorting: tokens assigned to random experts
  [t0→exp2, t1→exp0, t2→exp2, t3→exp1, ...]

AFTER sorting: tokens grouped by expert (contiguous in memory)
  [t1, t4, ... | t3, ... | t0, t2, ...]
   ←─expert0─→  ←─exp1─→  ←──expert2──→
```

Both approaches sort tokens by expert. The difference:
- **Sparse**: build index structures, call sparse GEMM that uses those indices
- **Direct**: use sort indices directly, compute weight column ranges on-the-fly

Same memory access pattern, less indirection, ~38% less overhead.

---

## SonicMoE Insights (December 2024)

Key optimizations from "SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations" (arXiv 2512.14080):

### 1. Gather Fusion is Critical
Fuse the gather operation into the GEMM prologue. Instead of:
```python
x_gathered = gather(x, indices)      # separate kernel, writes to GMEM
output = gemm(x_gathered, W)         # reads x_gathered from GMEM
```

Do:
```python
# Inside GEMM kernel, load x via indices directly (GMEM→SMEM with indirection)
x_tile = tl.load(x_ptr + indices[...] * stride + ...)
```

This eliminates a full read+write of the gathered tensor.

### 2. Don't Fuse Scatter (Counterintuitive!)
SonicMoE found that fusing scatter into the GEMM epilogue causes ~20% throughput degradation due to synchronous stores. Better to:
- Write GEMM output to a temporary buffer
- Run scatter as a separate kernel using TMA stores

### 3. Token Rounding (Optional Optimization)
Instead of strict dropless (pad to tile boundaries, waste compute), round token counts to exact tile multiples:

```
Dropless:        Expert 0 gets 137 tokens → pad to 256 → 46% wasted
Token rounding:  Expert 0 gets 128 tokens (adjust routing) → 0% wasted
```

Constraint: deviation from vanilla top-k is at most 1 tile per expert.
Result: ~16% higher TFLOPS, negligible quality loss.

**We'll implement dropless first**, but token rounding is worth considering for production.

### Architecture

```
Input x: [num_tokens, hidden_size]
           │
           ▼
┌─────────────────────┐
│   Router (Linear)   │  ← Standard nn.Linear, no custom kernel needed
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Top-K + Sorting    │  ← Kernel 1: sort tokens by expert
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Gather (Permute)  │  ← Kernel 2: gather tokens into expert-sorted order
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Grouped GEMM (Up)  │  ← Kernel 3: x @ W1, fused activation
│  + Activation       │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Grouped GEMM (Down) │  ← Kernel 4: intermediate @ W2
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Scatter + Weight   │  ← Kernel 5: scatter back, multiply by routing weights
└─────────────────────┘
           │
           ▼
Output: [num_tokens, hidden_size]
```

---

## Kernels to Implement

### Kernel 1: Sort Tokens by Expert

**What it does:** Given expert assignments `[num_tokens * top_k]`, produce sorted indices and token counts per expert.

**Inputs:**
- `selected_experts`: [num_tokens * top_k] int32 - which expert each (token, k) pair is assigned to
- `num_experts`: int

**Outputs:**
- `sorted_indices`: [num_tokens * top_k] int32 - permutation to sort by expert
- `tokens_per_expert`: [num_experts] int32 - histogram

**Difficulty: Easy-Medium**

You can use:
- `triton.sort` (if available in your Triton version)
- Radix sort implementation
- Or just call `torch.sort` on the expert IDs and keep track of indices

Megablocks uses a radix sort here (`ops.sort`). For a first pass, `torch.sort` works fine - it's not the bottleneck.

**Key insight:** You only need to sort by expert ID, not maintain stable order within experts. Radix sort with `ceil(log2(num_experts))` bits is fast.

---

### Kernel 2: Gather (Permute Tokens)

**What it does:** Rearrange tokens so all tokens for expert 0 are contiguous, then expert 1, etc.

**Inputs:**
- `x`: [num_tokens, hidden_size] - input activations
- `sorted_indices`: [num_tokens * top_k] - from kernel 1
- `tokens_per_expert`: [num_experts] - from kernel 1

**Outputs:**
- `x_gathered`: [total_expanded_tokens, hidden_size] - tokens in expert-sorted order

**Difficulty: Easy** (but consider fusing with GEMM—see below)

```python
@triton.jit
def gather_kernel(
    x_ptr,           # [num_tokens, hidden_size]
    indices_ptr,     # [total_expanded_tokens]
    output_ptr,      # [total_expanded_tokens, hidden_size]
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # which output row

    # Which original token does this output row come from?
    src_idx = tl.load(indices_ptr + pid)

    # Copy hidden_size elements
    for offset in range(0, hidden_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        vals = tl.load(x_ptr + src_idx * hidden_size + cols, mask=mask)
        tl.store(output_ptr + pid * hidden_size + cols, vals, mask=mask)
```

**Note:** This is memory-bound. The loop over hidden_size can be parallelized by launching a 2D grid (pid_row, pid_col). For large hidden_size, consider processing multiple columns per program.

**Optimization: Fuse with GEMM (SonicMoE insight)**

A separate gather kernel writes `x_gathered` to GMEM, then the GEMM reads it back. This is wasteful. Better to fuse gather into the GEMM prologue:

```python
# Instead of loading from pre-gathered buffer:
x_tile = tl.load(x_gathered_ptr + row * hidden + k_offset, ...)

# Load directly from original x using indices:
src_row = tl.load(indices_ptr + row)
x_tile = tl.load(x_ptr + src_row * hidden + k_offset, ...)
```

This eliminates a full tensor write+read. Implement standalone gather first for correctness, then fuse as optimization.

---

### Kernel 3: Grouped GEMM (Up-projection) + Fused Activation

**What it does:** For each expert, compute `activation(x_expert @ W1_expert)`.

This is the **main kernel** and the most complex.

**Inputs:**
- `x_gathered`: [total_tokens_padded, hidden_size] - gathered activations
- `W1`: [hidden_size, total_expert_width] - concatenated expert weights
- `expert_token_offsets`: [num_experts + 1] - cumulative token counts
- `expert_weight_offsets`: [num_experts + 1] - cumulative weight offsets
- `expert_widths`: [num_experts] - intermediate size per expert (for variable sizes)

**Outputs:**
- `intermediate`: [total_tokens_padded, total_expert_width] - but sparse! only valid regions filled

**Wait, output layout for variable experts?**

This is a key design decision. Options:

**Option A: Sparse output (match input token layout)**
```
Expert 0: tokens 0-99,   intermediate width 1024 → output rows 0-99,   cols 0-1023
Expert 1: tokens 100-149, intermediate width 512 → output rows 100-149, cols 0-511
```
Problem: each expert writes to different column ranges, messy for down-projection.

**Option B: Gathered output (concatenate expert outputs)**
```
Expert 0: 100 tokens × 1024 width = 102400 elements → output[0:102400]
Expert 1: 50 tokens × 512 width = 25600 elements → output[102400:128000]
```
Flattened, contiguous per expert. Need offset math but simpler memory access.

**Option C: Padded dense (waste memory but simpler)**
```
All experts use max_intermediate_size, waste space for smaller experts.
```

**Recommendation:** Option B for variable experts. Each expert's output is a contiguous chunk.

**Difficulty: Hard**

The kernel structure:

```python
@triton.jit
def grouped_gemm_up_kernel(
    # Pointers
    x_ptr, w1_ptr, out_ptr,
    # Expert metadata
    expert_token_offsets_ptr,    # [num_experts + 1]
    expert_weight_offsets_ptr,   # [num_experts + 1]
    expert_output_offsets_ptr,   # [num_experts + 1] - for Option B
    expert_widths_ptr,           # [num_experts]
    # Dimensions
    hidden_size,
    num_experts,
    # Tile sizes
    BLOCK_M: tl.constexpr,  # tokens tile
    BLOCK_N: tl.constexpr,  # intermediate tile
    BLOCK_K: tl.constexpr,  # hidden tile
):
    # Step 1: Figure out which expert and tile this program handles
    pid = tl.program_id(0)

    # This is the tricky part - need to map pid to (expert, m_tile, n_tile)
    # For variable experts, need prefix sum over tiles per expert
    expert_idx, m_tile, n_tile = decode_pid(pid, ...)

    # Step 2: Load expert-specific metadata
    token_start = tl.load(expert_token_offsets_ptr + expert_idx)
    token_end = tl.load(expert_token_offsets_ptr + expert_idx + 1)
    weight_start = tl.load(expert_weight_offsets_ptr + expert_idx)
    expert_width = tl.load(expert_widths_ptr + expert_idx)
    output_start = tl.load(expert_output_offsets_ptr + expert_idx)

    # Step 3: Compute tile boundaries
    m_start = token_start + m_tile * BLOCK_M
    m_end = min(m_start + BLOCK_M, token_end)
    n_start = n_tile * BLOCK_N
    n_end = min(n_start + BLOCK_N, expert_width)

    # Step 4: Standard tiled matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, hidden_size, BLOCK_K):
        # Load x tile: [BLOCK_M, BLOCK_K]
        # x is at x_ptr + (token_start + m_tile*BLOCK_M) * hidden_size + k
        x_tile = load_x_tile(x_ptr, m_start, k, ...)

        # Load w1 tile: [BLOCK_K, BLOCK_N]
        # w1 is at w1_ptr + k * total_width + (weight_start + n_start)
        w_tile = load_w_tile(w1_ptr, k, weight_start + n_start, ...)

        acc += tl.dot(x_tile, w_tile)

    # Step 5: Apply activation (fused)
    acc = activation(acc)  # silu, gelu, relu_squared, etc.

    # Step 6: Store output
    # For Option B: output at output_start + m_local * expert_width + n_start
    store_output(out_ptr, output_start, m_tile, n_start, acc, ...)
```

**The hard part: `decode_pid`**

For variable-sized experts, you need to map a linear program ID to (expert, m_tile, n_tile):

```python
def compute_tile_schedule(tokens_per_expert, expert_widths, BLOCK_M, BLOCK_N):
    """Precompute on CPU, pass to kernel."""
    tiles_per_expert = []
    for i in range(num_experts):
        m_tiles = ceil(tokens_per_expert[i] / BLOCK_M)
        n_tiles = ceil(expert_widths[i] / BLOCK_N)
        tiles_per_expert.append(m_tiles * n_tiles)

    # Prefix sum for pid decoding
    tile_offsets = [0] + cumsum(tiles_per_expert)
    total_tiles = tile_offsets[-1]

    return tile_offsets, total_tiles
```

In the kernel, use binary search or precomputed lookup to decode pid.

**SwiGLU complication:**

For SwiGLU, you compute `silu(x @ W_gate) * (x @ W_up)`. This means:
- W1 is actually `[hidden_size, 2 * intermediate_size]` per expert (gate + up concatenated)
- Or two separate weight matrices
- The kernel loads both, computes both matmuls, applies silu to one, multiplies

```python
# SwiGLU variant
gate = x @ W_gate  # [tokens, intermediate]
up = x @ W_up      # [tokens, intermediate]
intermediate = silu(gate) * up
```

Can fuse into one kernel by loading both weight tiles and computing in parallel.

---

### Kernel 4: Grouped GEMM (Down-projection)

**What it does:** For each expert, compute `intermediate_expert @ W2_expert`.

Very similar to Kernel 3, but:
- Input is the intermediate activations (variable width per expert)
- Output is hidden_size (uniform)
- No activation function (or maybe add residual here)

**Inputs:**
- `intermediate`: from kernel 3 (in Option B layout)
- `W2`: [total_expert_width, hidden_size] - concatenated
- Expert offset metadata

**Outputs:**
- `output_gathered`: [total_tokens_padded, hidden_size]

**Difficulty: Hard** (same as kernel 3, minus activation)

The structure mirrors kernel 3. Main difference:
- M dimension = tokens (same)
- K dimension = expert_width (variable per expert!)
- N dimension = hidden_size (uniform)

```python
for k in range(0, expert_width, BLOCK_K):  # K is variable now
    ...
```

This is actually slightly trickier because the K loop bound varies per expert.

---

### Kernel 5: Scatter + Weight Application

**What it does:** Un-permute tokens back to original order, multiply by routing weights, accumulate across top-k.

**Inputs:**
- `output_gathered`: [total_expanded_tokens, hidden_size]
- `sorted_indices`: [total_expanded_tokens] - the permutation from kernel 1
- `routing_weights`: [num_tokens, top_k] - the softmax/sigmoid weights from router

**Outputs:**
- `output`: [num_tokens, hidden_size]

**Difficulty: Medium**

The scatter is the inverse of gather. The tricky part: multiple experts contribute to each output token (top-k), so you need atomic adds or a smarter approach.

**Important: Don't fuse scatter with GEMM! (SonicMoE insight)**

Unlike gather fusion (which helps), fusing scatter into the GEMM epilogue causes ~20% throughput degradation due to synchronous store requirements. Keep scatter as a separate kernel.

**Option A: Atomic adds** (simple, use for initial implementation)
```python
@triton.jit
def scatter_kernel(...):
    pid = tl.program_id(0)  # which gathered row

    # Where does this go in the output?
    orig_idx = tl.load(indices_ptr + pid)
    token_idx = orig_idx // top_k
    k_idx = orig_idx % top_k

    weight = tl.load(routing_weights_ptr + token_idx * top_k + k_idx)

    for col in range(...):
        val = tl.load(gathered_ptr + pid * hidden_size + col)
        tl.atomic_add(output_ptr + token_idx * hidden_size + col, val * weight)
```

Atomics are slow but correct. Good enough for initial implementation.

**Option B: Process token-by-token (recommended for optimization)**

Launch one program per output token. Each program finds its top-k contributions, loads them, weights them, sums them. No atomics needed.

Requires precomputed reverse indices:
```python
# reverse_indices[token_idx, k] = position in gathered array
# Precompute after sorting, before scatter
```

```python
@triton.jit
def scatter_by_token_kernel(...):
    token_idx = tl.program_id(0)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for k in range(top_k):
        # Where is this token's k-th expert result in gathered array?
        gathered_idx = tl.load(reverse_indices_ptr + token_idx * top_k + k)
        weight = tl.load(routing_weights_ptr + token_idx * top_k + k)

        for col_offset in range(0, hidden_size, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            vals = tl.load(gathered_ptr + gathered_idx * hidden_size + cols, ...)
            # Accumulate weighted contribution
            acc += weight * vals
            tl.store(output_ptr + token_idx * hidden_size + cols, acc, ...)
            acc = tl.zeros(...)  # reset for next chunk
```

This is cleaner and faster than atomics.

**Option C: Sort by destination, segmented reduce**

Sort gathered outputs by destination token, then do a segmented reduction. More complex, similar performance to Option B.

---

## Backward Passes

For training, you need gradients. The MoE backward is more complex than forward.

### Backward Overview

Given `dL/d_output`, compute:
- `dL/d_input` (for backprop through the network)
- `dL/d_W1`, `dL/d_W2` (for weight updates)
- `dL/d_router` (for router weight updates)

### Backward through Scatter (Kernel 5 backward)

**Forward:** `output[i] = sum_k(weight[i,k] * gathered[perm[i,k]])`

**Backward:**
- `d_gathered[perm[i,k]] = weight[i,k] * d_output[i]` (gather pattern - easy)
- `d_weight[i,k] = gathered[perm[i,k]] · d_output[i]` (dot product for each k)

**Difficulty: Easy-Medium**

The gather in backward is simpler than scatter in forward (no atomics needed).

---

### Backward through Down-projection (Kernel 4 backward)

**Forward:** `Y = X @ W2` where X is intermediate, Y is pre-scatter output

**Backward:**
- `dX = dY @ W2.T` (for backprop) - another grouped GEMM
- `dW2 = X.T @ dY` (for weight update) - grouped GEMM with transposed inputs

**Difficulty: Hard**

Same complexity as forward GEMM kernels, but:
- `dX` computation: same structure as forward, different matrix
- `dW2` computation: outer product style, different tiling

The `dW2` kernel is trickier because you're accumulating over the token dimension:
- Each expert's `dW2` slice is `X_expert.T @ dY_expert`
- Shape: `[expert_width, hidden_size]`
- Need to reduce over tokens

---

### Backward through Activation (fused in Kernel 3 backward)

**SwiGLU backward:**
```
Forward: out = silu(gate) * up
Backward:
  d_gate = d_out * up * silu'(gate)
  d_up = d_out * silu(gate)
```

Where `silu'(x) = silu(x) + sigmoid(x) * (1 - silu(x))` = `sigmoid(x) * (1 + x * (1 - sigmoid(x)))`

This gets fused into the up-projection backward.

**Difficulty: Medium** (just math, fuses into existing kernel)

---

### Backward through Up-projection (Kernel 3 backward)

**Forward:** `intermediate = activation(X @ W1)`

**Backward:**
- `dX = d_intermediate @ W1.T` - grouped GEMM
- `dW1 = X.T @ d_intermediate` - grouped GEMM, reduction over tokens

Plus the activation backward (fused).

**Difficulty: Hard**

---

### Backward through Gather (Kernel 2 backward)

**Forward:** `gathered = x[indices]` (gather)

**Backward:** `d_x = scatter_add(d_gathered, indices)` (scatter-add)

Multiple gathered positions may come from the same input token (top-k), so need to accumulate.

**Difficulty: Medium**

Same atomic add issue as forward scatter, but now in backward.

---

### Backward through Router

**Forward:** `logits = x @ W_router`, then top-k selection

**Backward:**
- `dW_router = x.T @ d_logits` - standard linear backward
- `d_x += d_logits @ W_router.T` - add to input gradient

The tricky part: what is `d_logits`?

The routing weights affect the output through the weighted scatter. So:
```
d_routing_weight[i,k] = gathered[perm[i,k]] · d_output[i]  (from scatter backward)
```

Then through softmax/sigmoid:
```
d_logits = d_routing_weight * softmax_backward(...)
```

For sigmoid routing (what you're using):
```
d_logits = d_routing_weight * sigmoid(logits) * (1 - sigmoid(logits))
```

**Difficulty: Medium**

---

## Summary: Difficulty Ratings

| Kernel | Forward | Backward | Notes |
|--------|---------|----------|-------|
| Sort | Easy | N/A (no gradient) | Use torch.sort or simple radix |
| Gather | Easy | Medium | Forward easy, backward has scatter-add |
| Grouped GEMM Up | **Hard** | **Hard** | Main complexity, variable expert scheduling |
| Grouped GEMM Down | **Hard** | **Hard** | Similar to up-projection |
| Scatter | Medium | Easy | Atomics or clever indexing |
| Router | Easy | Medium | Standard linear + softmax/sigmoid backward |

---

## Implementation Order Recommendation

1. **Start with uniform experts** - remove variable sizing initially
2. **Forward pass first (unfused):**
   - Kernel 2 (gather) - simple, build intuition
   - Kernel 3 (grouped GEMM up) - the hard one, spend time here
   - Kernel 4 (grouped GEMM down) - similar to 3
   - Kernel 5 (scatter) - use atomics initially
3. **Test forward correctness** against your current stk implementation
4. **Backward passes:**
   - Work backwards: scatter grad → down grad → activation grad → up grad → gather grad
   - Test each against `torch.autograd.gradcheck`
5. **Add variable expert support:**
   - Update tile scheduler for variable widths
   - Update offset calculations
6. **Optimize (in this order):**
   - **Fuse gather into GEMM up** - biggest win, eliminates GMEM round-trip
   - Replace atomic scatter with token-by-token approach
   - Tune tile sizes (BLOCK_M, BLOCK_N, BLOCK_K)
   - Consider token rounding if padding waste is significant
   - **Don't fuse scatter** - keep it separate (SonicMoE finding)

---

## Key Challenges

### 1. Tile Scheduling for Variable Experts

The grouped GEMM needs to map `program_id → (expert, m_tile, n_tile)`. With variable token counts AND variable expert widths, this mapping is non-trivial.

**Solution:** Precompute a tile schedule on CPU:
```python
# For each expert, compute number of tiles
# Build prefix sum for fast pid decoding
# Pass schedule to kernel
```

### 2. Memory Layout Decisions

How to lay out the intermediate activations when experts have different widths?

**Recommendation:** Concatenated layout (Option B above). Each expert's output is a contiguous chunk. Track offsets.

### 3. Atomic Operations in Scatter

Multiple top-k contributions accumulate into each output token.

**Solutions:**
- Atomics (simple, slower)
- Sort by destination then segmented reduce (faster, more complex)
- Deterministic ordering with precomputed reverse indices

### 4. Numerical Stability

- Use float32 accumulation in matmuls, cast to bf16 for storage
- Be careful with softmax/sigmoid in routing (numerical stability tricks)

### 5. Autograd Integration

Wrap everything in `torch.autograd.Function`:
```python
class GroupedMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, router_weights, ...):
        # Run forward kernels
        ctx.save_for_backward(x, w1, w2, intermediate, ...)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Run backward kernels
        return grad_x, grad_w1, grad_w2, grad_router, ...
```

---

## Resources

- **Triton tutorials:** https://triton-lang.org/main/getting-started/tutorials/
- **Triton matmul tutorial:** Start here, understand tiling
- **SonicMoE paper:** arXiv 2512.14080 - IO-aware insights
- **Megablocks paper:** Understanding block-sparse approach (what you're replacing)
- **FlashAttention paper:** Similar IO-aware optimization philosophy

---

## Testing Strategy

```python
def test_forward_correctness():
    """Compare Triton impl against reference (your current stk impl)."""
    x = torch.randn(...)

    # Reference
    out_ref, _ = moe_reference(x)

    # Triton
    out_triton, _ = moe_triton(x)

    torch.testing.assert_close(out_ref, out_triton, rtol=1e-2, atol=1e-2)

def test_backward_correctness():
    """Use gradcheck."""
    x = torch.randn(..., requires_grad=True)
    torch.autograd.gradcheck(moe_triton, x, eps=1e-3, atol=1e-2, rtol=1e-2)
```

---

## Small-Scale Considerations

Standard Triton tile sizes (128x128) are tuned for large-batch training on big models. At smaller scales (small batches, many experts, narrow intermediate sizes), these may not be optimal.

### The Problem

With small batches and many experts, tokens-per-expert can be low:
```
batch=4, seq=512, top_k=2, experts=8
total_assignments = 4 * 512 * 2 = 4096
average_per_expert ≈ 512

But with imbalanced routing, some expert might get only 100 tokens:
  BLOCK_M=128 → pad to 128 → 28% padding waste
  BLOCK_M=32  → pad to 128 → only 6% waste (rounds to 4 tiles of 32)
```

Similarly, narrow experts (256-width intermediate) don't need large BLOCK_N.

### Tile Size Tuning

Triton's autotune makes this easy to explore:

```python
@triton.autotune(
    configs=[
        # Standard sizes
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        # Smaller M for few-tokens-per-expert
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}),
        # Non-square for narrow experts
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}),
    ],
    key=['total_tokens', 'hidden_size', 'max_expert_width'],
)
@triton.jit
def grouped_gemm_kernel(...):
    ...
```

### What to Try

| Scenario | Consider |
|----------|----------|
| Few tokens per expert (<128) | Smaller BLOCK_M (32, 64) |
| Narrow experts (256-512 width) | Smaller BLOCK_N (32, 64) |
| Very small hidden_size | Smaller BLOCK_K (16, 32) |
| Large batches | Standard sizes (64, 128) often best |

### Tradeoffs

- **Smaller tiles**: Less padding waste, but more tiles to schedule, potentially worse tensor core utilization
- **Larger tiles**: Better hardware efficiency, but more padding waste at small scales
- **Non-square tiles**: Can match your actual dimensions better (e.g., BLOCK_M=32, BLOCK_N=64 for few tokens but moderate width)

This is a key advantage of custom Triton over library code—tune for your actual workload. Profile with your real batch sizes and expert configurations.

---

Good luck! Start with the grouped GEMM forward - once you nail that, everything else builds on the same patterns.
