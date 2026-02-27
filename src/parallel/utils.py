"""Parallel processing utilities for HPO batch execution."""

from __future__ import annotations

import queue
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

import optuna


ResultItem = Tuple[int, Any]

# How long (seconds) to wait on result_queue.get() before checking worker health.
_RESULT_POLL_INTERVAL = 30.0
# How long (seconds) since a worker last produced a result before we call it dead.
_WORKER_TIMEOUT = 1800.0  # 30 minutes


def aggregate_results(results: Iterable[ResultItem]) -> List[Any]:
    """Return results sorted by sample index."""
    sorted_results = sorted(results, key=lambda item: item[0])
    return [result for _, result in sorted_results]


def run_batch_parallel(
    samples: Iterable[ResultItem],
    processor,
    trial: Optional[optuna.Trial] = None,
    prune_after_n: int = 10,
    intermediate_score_fn: Optional[Callable[[List[ResultItem]], float]] = None,
    worker_timeout: float = _WORKER_TIMEOUT,
) -> List[Any]:
    """Submit work to a processor and collect results.

    Args:
        samples: Iterable of (index, payload) items to submit.
        processor: Parallel processor with submit() and result_queue.
        trial: Optional Optuna trial for pruning.
        prune_after_n: Number of results before checking pruning.
        intermediate_score_fn: Computes intermediate score for pruning.
        worker_timeout: Seconds without any result before assuming workers are
            dead and injecting error placeholders for all pending samples.
            Default: 1800 s (30 min).  Set to 0 to disable (original behaviour).
    """
    sample_list = list(samples)
    n_total = len(sample_list)
    submitted_indices = {idx for idx, _ in sample_list}

    for idx, payload in sample_list:
        processor.submit(idx, payload)

    results: List[ResultItem] = []
    last_result_time = time.monotonic()

    while len(results) < n_total:
        try:
            idx, result = processor.result_queue.get(timeout=_RESULT_POLL_INTERVAL)
            results.append((idx, result))
            last_result_time = time.monotonic()
        except queue.Empty:
            # No result yet — check whether any worker is still alive.
            if worker_timeout > 0:
                elapsed = time.monotonic() - last_result_time
                if elapsed > worker_timeout:
                    # Workers appear dead; inject error placeholders for remaining samples.
                    collected_indices = {i for i, _ in results}
                    missing = submitted_indices - collected_indices
                    print(
                        f"[run_batch_parallel] WARNING: no result received for "
                        f"{elapsed:.0f}s. Treating {len(missing)} pending "
                        f"sample(s) as failed: {sorted(missing)}"
                    )
                    for m_idx in sorted(missing):
                        results.append((m_idx, {"error": "worker_timeout", "converged": False}))
                    break
            # Check if all worker processes are dead (handles crash without timeout).
            if hasattr(processor, "processes") and processor.processes:
                alive = any(p.is_alive() for p in processor.processes)
                if not alive:
                    collected_indices = {i for i, _ in results}
                    missing = submitted_indices - collected_indices
                    if missing:
                        print(
                            f"[run_batch_parallel] WARNING: all worker processes "
                            f"are dead. Injecting error for {len(missing)} "
                            f"pending sample(s): {sorted(missing)}"
                        )
                        for m_idx in sorted(missing):
                            results.append(
                                (m_idx, {"error": "worker_process_dead", "converged": False})
                            )
                    break
            continue

        if (
            trial is not None
            and intermediate_score_fn is not None
            and len(results) == prune_after_n
        ):
            score = intermediate_score_fn(results)
            trial.report(score, step=prune_after_n)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return aggregate_results(results)
