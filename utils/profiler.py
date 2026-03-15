'''
PyTorch profiler wrapper for CPU profiling(runs on Macbook M2 CPU).
'''

import logging
import time
from contextlib import contextmanager
from typing import Dict

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

logger = logging.getLogger(__name__)


@contextmanager
def profile_model(model, batch: Dict, n_warmup: int = 3, n_active: int = 5):
    # CPU-only
    activities = [ProfilerActivity.CPU]

    # Remove label keys that are not model inputs
    input_batch = {k: v for k, v in batch.items()
                   if k not in ("click", "purchase", "ctcvr")}

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1, warmup=n_warmup, active=n_active),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(1 + n_warmup + n_active):
            with record_function("model_forward"):
                with torch.no_grad():
                    _ = model(input_batch)
            prof.step()
        yield prof


def print_profiler_summary(prof, top_k: int = 15):
    # Print top-K operations sorted by CPU time.
    print("\n" + "=" * 80)
    print(f"PyTorch Profiler — Top {top_k} ops by CPU time (Apple M2)")
    print("=" * 80)
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=top_k,
        )
    )
    print("=" * 80 + "\n")


def measure_inference_latency(
    model,
    batch: Dict,
    n_runs:  int = 100,
    warmup:  int = 10,
) -> Dict:
    # Measure mean / P50 / P95 / P99 inference latency on CPU (milliseconds).
    model.eval()
    input_batch = {k: v for k, v in batch.items()
                   if k not in ("click", "purchase", "ctcvr")}

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_batch)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(input_batch)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    result = {
        "mean_ms": float(latencies.mean()),
        "p50_ms":  float(np.percentile(latencies, 50)),
        "p95_ms":  float(np.percentile(latencies, 95)),
        "p99_ms":  float(np.percentile(latencies, 99)),
        "min_ms":  float(latencies.min()),
        "max_ms":  float(latencies.max()),
    }
    logger.info(
        f"CPU inference latency (batch_size={next(iter(input_batch.values())).shape[0]}) | "
        f"mean={result['mean_ms']:.2f}ms | "
        f"p50={result['p50_ms']:.2f}ms | "
        f"p95={result['p95_ms']:.2f}ms | "
        f"p99={result['p99_ms']:.2f}ms"
    )
    return result


def run_profiling_report(model, batch: Dict, top_k: int = 15) -> Dict:
    # Profile operator breakdown
    with profile_model(model, batch) as prof:
        pass
    print_profiler_summary(prof, top_k=top_k)

    # Latency measurement
    stats = measure_inference_latency(model, batch)
    return stats