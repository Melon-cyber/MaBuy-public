import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("PipelineTracer")


class ExecutionTracer:
    """
    Zero-Overhead Microsecond Execution Profiler for trading and prediction pipelines.
    Allows pinpointing exact latency bottlenecks (DataFetch, FeatureCalc, Inference, Order Execution, DB Commit).
    """

    def __init__(self, trace_name: str, warn_threshold_ms: float = 100.0):
        self.trace_name = trace_name
        self.warn_threshold_ms = warn_threshold_ms
        self.stages: List[Tuple[str, float]] = []
        self._start_total = time.perf_counter()

    @contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stages.append((name, elapsed_ms))

    def finish(self) -> Dict[str, Any]:
        total_ms = (time.perf_counter() - self._start_total) * 1000.0
        breakdown_str = " | ".join([f"{name}: {t:.1f}ms" for name, t in self.stages])

        if total_ms > self.warn_threshold_ms:
            logger.warning(
                f"⚠️ [SLOW PIPELINE] {self.trace_name} => Total: {total_ms:.1f}ms [{breakdown_str}]"
            )
        else:
            logger.debug(
                f"⚡ [FAST PIPELINE] {self.trace_name} => Total: {total_ms:.1f}ms [{breakdown_str}]"
            )

        return {
            "trace_name": self.trace_name,
            "total_ms": total_ms,
            "stages": dict(self.stages),
        }
