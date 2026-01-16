# HPO Parallelization Plan

## Goal
Improve hardware utilization from ~17-20% GPU / ~1% of 192 CPU cores to ~80%+ for both HIP and SCINE HPO workloads.

## Key Constraints

**HIP (PyTorch CUDA):**
- Cannot pickle models across processes
- Must load checkpoint separately in each worker via `torch.multiprocessing.spawn`
- Workers can share GPU via CUDA context

**SCINE:**
- Uses singleton ModuleManager (global state)
- Must create fresh calculator instance per worker
- Must set OMP_NUM_THREADS BEFORE importing SCINE

## Architecture

```
Main Process (Optuna)
    │
    ├── Spawns N Workers (torch.multiprocessing.spawn for HIP / mp.spawn for SCINE)
    │       │
    │       ├── Worker 0: Load model/calculator, process samples
    │       ├── Worker 1: Load model/calculator, process samples
    │       └── Worker N: ...
    │
    └── Collects results via mp.Queue, aggregates, reports to Optuna
```

## Files to Create

### 1. `src/parallel/__init__.py`
Exports for the parallel processing module.

### 2. `src/parallel/hip_parallel.py`
- `hip_worker_fn()` - Worker init function (loads model from checkpoint)
- `ParallelHIPProcessor` class - Manages worker pool, work/result queues
- Uses `torch.multiprocessing.spawn` start method

### 3. `src/parallel/scine_parallel.py`
- `scine_worker_fn()` - Worker init (sets OMP threads, creates calculator)
- `ParallelSCINEProcessor` class - Same pattern as HIP

### 4. `src/parallel/utils.py`
- `run_batch_parallel()` - Generic parallel batch runner
- `aggregate_results()` - Combines worker results

### 5. `src/noisy/hip_multi_mode_eckartmw_hpo_parallel.py`
- New parallel version with `--n-workers` argument (default: 4)
- Uses `ParallelHIPProcessor` for sample processing
- Copies structure from existing script

### 6. `src/noisy/scine_multi_mode_eckartmw_hpo_parallel.py`
- New parallel version with `--n-workers` (default: 16)
- `--threads-per-worker` argument (default: auto = 192/n_workers)
- Uses `ParallelSCINEProcessor`

### 7. `src/experiments/Sella/hip_sella_hpo_parallel.py`
- New parallel version with `--n-workers` (default: 4)
- Uses `ParallelHIPProcessor` for sample processing

### 8. `src/experiments/Sella/scine_sella_hpo_parallel.py`
- New parallel version with `--n-workers` (default: 16)
- Uses `ParallelSCINEProcessor`

### 9. New SLURM Scripts (4 files)
New parallel-specific SLURM scripts:
- `scripts/Trillium/noisy/hip_multi_mode_eckartmw_hpo_parallel.slurm`
- `scripts/Trillium/noisy/scine_multi_mode_eckartmw_hpo_parallel.slurm`
- `scripts/Trillium/experiments/Sella/hip_sella_hpo_parallel.slurm`
- `scripts/Trillium/experiments/Sella/scine_sella_hpo_parallel.slurm`

## Files Unchanged
Original sequential scripts remain untouched for backward compatibility:
- `src/noisy/hip_multi_mode_eckartmw_hpo.py`
- `src/noisy/scine_multi_mode_eckartmw_hpo.py`
- `src/experiments/Sella/hip_sella_hpo.py`
- `src/experiments/Sella/scine_sella_hpo_bayesian.py`

## Implementation Details

### HIP Worker Pattern
```python
def hip_worker_fn(rank, checkpoint_path, device, work_queue, result_queue, done_event):
    # Load model in this process
    calculator = EquiformerTorchCalculator(checkpoint_path=checkpoint_path)
    predict_fn = make_predict_fn_from_calculator(calculator, "hip")

    while not done_event.is_set():
        work = work_queue.get()
        if work is None: break
        result = process_sample(predict_fn, work)
        result_queue.put(result)
```

### SCINE Worker Pattern
```python
def scine_worker_fn(rank, functional, threads, work_queue, result_queue, done_event):
    # Set threads BEFORE import
    os.environ["OMP_NUM_THREADS"] = str(threads)
    # Now import and create calculator
    calculator = ScineSparrowCalculator(functional=functional)
    predict_fn = make_predict_fn_from_calculator(calculator, "scine")

    while not done_event.is_set():
        work = work_queue.get()
        if work is None: break
        result = process_sample(predict_fn, work)
        result_queue.put(result)
```

### Parallel Batch Processing
```python
def run_batch_parallel(samples, params, processor, trial=None, prune_after_n=10):
    # Submit work
    for i, sample in enumerate(samples):
        processor.work_queue.put((i, sample, params))

    # Collect results with pruning support
    results = []
    for _ in range(len(samples)):
        idx, result = processor.result_queue.get()
        results.append((idx, result))

        # Check pruning after N samples
        if trial and len(results) == prune_after_n:
            score = compute_intermediate_score(results)
            trial.report(score, step=prune_after_n)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return aggregate_results(sorted(results))
```

## Configuration

### HIP (H100 + 24 cores)
- `--n-workers 4-8` (recommend starting with 4)
- Each worker: ~500MB VRAM, shares GPU via CUDA context
- `OMP_NUM_THREADS=4` per worker for CPU ops

### SCINE (192 cores)
- `--n-workers 16` (for DFTB0)
- `--threads-per-worker 12` (192/16 = 12)
- Each worker creates fresh calculator

## Expected Results

| Workload | Before | After | Speedup |
|----------|--------|-------|---------|
| HIP GPU Util | 17-20% | 60-80% | 3-4x |
| HIP Trial Time | ~7 min | ~2 min | 3-4x |
| SCINE Core Util | <10% | 80%+ | 8-16x |
| SCINE Trial Time | ~20 min | ~2 min | 8-10x |

## Verification Steps

1. Run HIP HPO with `--n-workers 1` (baseline) then `--n-workers 4`
2. Check `nvidia-smi` shows increased GPU utilization
3. Run SCINE HPO with `--n-workers 1` then `--n-workers 16`
4. Check `htop` shows balanced core usage
5. Verify Optuna pruning still works (intermediate values reported)
6. Verify results match sequential version (same final metrics)
