#!/bin/bash
# =============================================================================
# Sync HPO Results to W&B (Run from Trillium login node)
# =============================================================================
#
# This script syncs:
#   1. Offline W&B runs from compute jobs
#   2. SQLite databases as W&B Artifacts
#
# Usage (from login node):
#   ./scripts/sync_hpo_results.sh                    # Sync all
#   ./scripts/sync_hpo_results.sh --dbs-only         # Only upload DBs
#   ./scripts/sync_hpo_results.sh --runs-only        # Only sync W&B runs
#   ./scripts/sync_hpo_results.sh --job 201956       # Sync specific job
#
# Prerequisites:
#   - Run from Trillium login node (has internet)
#   - W&B logged in: wandb login
#   - Python venv activated
# =============================================================================

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-/project/rrg-aspuru/memoozd/ts-tools}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/memoozd/ts-tools-scratch}"
DB_DIR="${SCRATCH_DIR}/dbs"
WANDB_OFFLINE_DIR="${SCRATCH_DIR}/hpo"

# W&B settings (match your HPO scripts)
WANDB_ENTITY="${WANDB_ENTITY:-memo-ozdincer-university-of-toronto}"

# Parse arguments
SYNC_DBS=true
SYNC_RUNS=true
SPECIFIC_JOB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dbs-only)
            SYNC_RUNS=false
            shift
            ;;
        --runs-only)
            SYNC_DBS=false
            shift
            ;;
        --job)
            SPECIFIC_JOB="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "HPO Results Sync to W&B"
echo "=============================================="
echo "Date: $(date)"
echo "DB Dir: $DB_DIR"
echo "W&B Offline Dir: $WANDB_OFFLINE_DIR"
echo "Sync DBs: $SYNC_DBS"
echo "Sync Runs: $SYNC_RUNS"
echo "Specific Job: ${SPECIFIC_JOB:-<all>}"
echo "=============================================="

# Activate environment
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# =============================================================================
# 1. Sync Offline W&B Runs
# =============================================================================
if [[ "$SYNC_RUNS" == "true" ]]; then
    echo ""
    echo "=== Syncing Offline W&B Runs ==="

    # Find offline runs
    if [[ -n "$SPECIFIC_JOB" ]]; then
        OFFLINE_DIRS=$(find "$WANDB_OFFLINE_DIR" -path "*${SPECIFIC_JOB}*/wandb/offline-run-*" -type d 2>/dev/null || true)
    else
        OFFLINE_DIRS=$(find "$WANDB_OFFLINE_DIR" -path "*/wandb/offline-run-*" -type d 2>/dev/null || true)
    fi

    if [[ -z "$OFFLINE_DIRS" ]]; then
        echo "No offline runs found to sync"
    else
        echo "Found offline runs:"
        echo "$OFFLINE_DIRS" | head -10

        for dir in $OFFLINE_DIRS; do
            echo ""
            echo "Syncing: $dir"
            wandb sync "$dir" || echo "[WARN] Failed to sync $dir"
        done
    fi
fi

# =============================================================================
# 2. Upload SQLite DBs as W&B Artifacts
# =============================================================================
if [[ "$SYNC_DBS" == "true" ]]; then
    echo ""
    echo "=== Uploading SQLite DBs as W&B Artifacts ==="

    # Find DBs
    if [[ -n "$SPECIFIC_JOB" ]]; then
        DB_FILES=$(find "$DB_DIR" -name "*${SPECIFIC_JOB}*.db" 2>/dev/null || true)
    else
        DB_FILES=$(find "$DB_DIR" -name "*.db" 2>/dev/null || true)
    fi

    if [[ -z "$DB_FILES" ]]; then
        echo "No SQLite databases found in $DB_DIR"
    else
        echo "Found databases:"
        echo "$DB_FILES"

        # Upload each DB as an artifact
        python3 << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    sys.exit(1)

db_dir = os.environ.get("DB_DIR", "/scratch/memoozd/ts-tools-scratch/dbs")
entity = os.environ.get("WANDB_ENTITY", "memo-ozdincer-university-of-toronto")
specific_job = os.environ.get("SPECIFIC_JOB", "")

# Mapping of DB patterns to projects
PROJECT_MAP = {
    "hip_sella": "sella-hpo",
    "scine_sella": "sella-hpo",
    "hip_multi_mode": "hip-multi-mode-hpo",
    "scine_multi_mode": "scine-multi-mode-hpo",
}

db_files = list(Path(db_dir).glob("*.db"))
if specific_job:
    db_files = [f for f in db_files if specific_job in f.name]

if not db_files:
    print("No databases to upload")
    sys.exit(0)

print(f"\nUploading {len(db_files)} database(s) as W&B artifacts...")

for db_path in db_files:
    db_name = db_path.stem

    # Determine project from filename
    project = "hpo-databases"  # default
    for pattern, proj in PROJECT_MAP.items():
        if pattern in db_name:
            project = proj
            break

    print(f"\n  DB: {db_path.name}")
    print(f"  Project: {project}")
    print(f"  Artifact: optuna-study/{db_name}")

    try:
        # Initialize a run just for uploading the artifact
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="upload-db",
            name=f"db-upload-{db_name}",
            tags=["db-upload", "optuna", db_name],
        )

        # Create artifact
        artifact = wandb.Artifact(
            name=db_name,
            type="optuna-study",
            description=f"Optuna SQLite database for HPO study: {db_name}",
            metadata={
                "study_name": db_name,
                "file_size_mb": db_path.stat().st_size / (1024 * 1024),
            }
        )
        artifact.add_file(str(db_path))

        # Log and finish
        run.log_artifact(artifact)
        run.finish()

        print(f"  ✓ Uploaded successfully")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n=== Upload Complete ===")
PYTHON_SCRIPT
    fi
fi

echo ""
echo "=============================================="
echo "Sync complete at $(date)"
echo "=============================================="
echo ""
echo "View results:"
echo "  https://wandb.ai/$WANDB_ENTITY"
echo ""
echo "Download artifacts locally:"
echo "  wandb artifact get $WANDB_ENTITY/sella-hpo/<artifact-name>:latest"
echo "=============================================="
