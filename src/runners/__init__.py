"""CLI/entrypoint runners for experiments.

These are intentionally thin orchestration layers around:
- `src/core_algos/*` (algorithm logic)
- `src/dependencies/*` (calculator adapters + Hessian projection)
- `src/logging/*` (plots + W&B)

They exist so cluster SLURM scripts can call stable `python -m src.runners.*` modules
without importing the large legacy scripts directly.
"""
