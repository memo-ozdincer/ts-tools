"""HIP parallel worker pool for sample processing."""

from __future__ import annotations

import inspect
import queue
from typing import Any, Callable, Dict, Optional

import torch
import torch.multiprocessing as mp

from ..runners._predict import make_predict_fn_from_calculator
from ..dependencies.common_utils import EquiformerTorchCalculator


def hip_worker_fn(
    rank: int,
    checkpoint_path: str,
    device: str,
    work_queue,
    result_queue,
    done_event,
    worker_fn: Callable[..., Any],
    worker_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Worker loop: load HIP calculator and process samples."""
    torch.set_grad_enabled(False)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(0)

    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )
    predict_fn = make_predict_fn_from_calculator(calculator, "hip")

    kwargs = worker_kwargs or {}
    signature = inspect.signature(worker_fn)
    use_calculator = "calculator" in signature.parameters
    while not done_event.is_set():
        try:
            item = work_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        idx, payload = item
        try:
            if use_calculator:
                result = worker_fn(predict_fn, calculator, payload, **kwargs)
            else:
                result = worker_fn(predict_fn, payload, **kwargs)
        except Exception as exc:
            result = {"error": str(exc)}
        result_queue.put((idx, result))


class ParallelHIPProcessor:
    """Parallel processing pool for HIP workloads."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        n_workers: int,
        worker_fn: Callable[..., Any],
        worker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_workers = n_workers
        self.worker_fn = worker_fn
        self.worker_kwargs = worker_kwargs or {}

        self.ctx = mp.get_context("spawn")
        self.work_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.done_event = self.ctx.Event()
        self.processes = []

    def start(self) -> None:
        if self.processes:
            return
        for rank in range(self.n_workers):
            proc = self.ctx.Process(
                target=hip_worker_fn,
                args=(
                    rank,
                    self.checkpoint_path,
                    self.device,
                    self.work_queue,
                    self.result_queue,
                    self.done_event,
                    self.worker_fn,
                    self.worker_kwargs,
                ),
            )
            proc.start()
            self.processes.append(proc)

    def submit(self, idx: int, payload: Any) -> None:
        self.work_queue.put((idx, payload))

    def close(self) -> None:
        if not self.processes:
            return
        self.done_event.set()
        for _ in range(self.n_workers):
            self.work_queue.put(None)
        for proc in self.processes:
            proc.join()
        self.processes = []
