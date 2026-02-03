"""CUDA event-based step profiler for MoE operations."""

import torch
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StepTiming:
    """Timing result for a single step."""
    name: str
    time_ms: float


@dataclass
class ProfileResult:
    """Results from a profiled forward pass."""
    step_times: list[StepTiming] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(s.time_ms for s in self.step_times)

    def as_dict(self) -> dict[str, float]:
        return {s.name: s.time_ms for s in self.step_times}

    def print_breakdown(self, indent: str = "  "):
        """Print timing breakdown with percentages."""
        total = self.total_ms
        for step in self.step_times:
            pct = (step.time_ms / total * 100) if total > 0 else 0
            print(f"{indent}{step.name:<25} {step.time_ms:>8.3f} ms ({pct:>5.1f}%)")
        print(f"{indent}{'TOTAL':<25} {total:>8.3f} ms")


class CUDAStepProfiler:
    """Profile individual steps using CUDA events.

    Usage:
        profiler = CUDAStepProfiler()

        with profiler.step("routing"):
            logits = router(x)

        with profiler.step("gather"):
            x_gathered = gather(x, indices)

        result = profiler.get_result()
        result.print_breakdown()
    """

    def __init__(self):
        self._steps: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._current_step: str | None = None

    def reset(self):
        """Clear all recorded steps."""
        self._steps = []
        self._current_step = None

    @contextmanager
    def step(self, name: str):
        """Context manager to time a step."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        try:
            yield
        finally:
            end.record()
            self._steps.append((name, start, end))

    def get_result(self) -> ProfileResult:
        """Get profiling results. Synchronizes CUDA."""
        torch.cuda.synchronize()

        step_times = []
        for name, start, end in self._steps:
            time_ms = start.elapsed_time(end)
            step_times.append(StepTiming(name=name, time_ms=time_ms))

        return ProfileResult(step_times=step_times)


class AggregateProfiler:
    """Aggregate profiling results across multiple runs.

    Usage:
        agg = AggregateProfiler()

        for _ in range(num_runs):
            profiler = CUDAStepProfiler()
            # ... run profiled forward ...
            agg.add_result(profiler.get_result())

        agg.print_summary()
    """

    def __init__(self):
        self._results: list[ProfileResult] = []

    def add_result(self, result: ProfileResult):
        self._results.append(result)

    @property
    def num_runs(self) -> int:
        return len(self._results)

    def get_mean_times(self) -> dict[str, float]:
        """Get mean time for each step across all runs."""
        if not self._results:
            return {}

        step_times: dict[str, list[float]] = defaultdict(list)
        for result in self._results:
            for step in result.step_times:
                step_times[step.name].append(step.time_ms)

        return {name: sum(times) / len(times) for name, times in step_times.items()}

    def get_std_times(self) -> dict[str, float]:
        """Get standard deviation for each step."""
        if not self._results:
            return {}

        means = self.get_mean_times()
        step_times: dict[str, list[float]] = defaultdict(list)
        for result in self._results:
            for step in result.step_times:
                step_times[step.name].append(step.time_ms)

        stds = {}
        for name, times in step_times.items():
            mean = means[name]
            variance = sum((t - mean) ** 2 for t in times) / len(times)
            stds[name] = variance ** 0.5
        return stds

    def print_summary(self, indent: str = "  "):
        """Print summary with mean times and percentages."""
        means = self.get_mean_times()
        stds = self.get_std_times()

        if not means:
            print(f"{indent}No profiling data")
            return

        total = sum(means.values())

        # Get step order from first result
        step_order = [s.name for s in self._results[0].step_times]

        print(f"{indent}Step Profiling ({self.num_runs} runs):")
        print(f"{indent}{'-' * 55}")
        for name in step_order:
            if name in means:
                pct = (means[name] / total * 100) if total > 0 else 0
                print(f"{indent}  {name:<22} {means[name]:>7.3f} +/- {stds[name]:>6.3f} ms ({pct:>5.1f}%)")
        print(f"{indent}{'-' * 55}")
        total_std = (sum(s ** 2 for s in stds.values())) ** 0.5  # Approximate
        print(f"{indent}  {'TOTAL':<22} {total:>7.3f} +/- {total_std:>6.3f} ms")
