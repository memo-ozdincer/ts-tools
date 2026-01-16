"""Parallel processing utilities for HPO batch execution."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple

import optuna


ResultItem = Tuple[int, Any]


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
) -> List[Any]:
    """Submit work to a processor and collect results.

    Args:
        samples: Iterable of (index, payload) items to submit.
        processor: Parallel processor with submit() and result_queue.
        trial: Optional Optuna trial for pruning.
        prune_after_n: Number of results before checking pruning.
        intermediate_score_fn: Computes intermediate score for pruning.
    """
    sample_list = list(samples)
    for idx, payload in sample_list:
        processor.submit(idx, payload)

    results: List[ResultItem] = []
    for _ in range(len(sample_list)):
        idx, result = processor.result_queue.get()
        results.append((idx, result))

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
