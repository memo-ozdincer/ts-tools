#!/usr/bin/env python3
"""
Download HPO SQLite databases from W&B Artifacts.

Usage:
    # List available artifacts
    python scripts/download_hpo_artifacts.py --list

    # Download specific artifact
    python scripts/download_hpo_artifacts.py --name scine_sella_hpo_job833394

    # Download all artifacts to local directory
    python scripts/download_hpo_artifacts.py --all --output-dir ./hpo_results

    # Download and launch Optuna dashboard
    python scripts/download_hpo_artifacts.py --name scine_sella_hpo_job833394 --dashboard
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download HPO artifacts from W&B")
    parser.add_argument("--entity", default="memo-ozdincer-university-of-toronto")
    parser.add_argument("--project", default="sella-hpo", help="W&B project name")
    parser.add_argument("--list", action="store_true", help="List available artifacts")
    parser.add_argument("--name", type=str, help="Artifact name to download")
    parser.add_argument("--all", action="store_true", help="Download all artifacts")
    parser.add_argument("--output-dir", type=str, default="./hpo_downloads")
    parser.add_argument("--dashboard", action="store_true", help="Launch Optuna dashboard after download")
    parser.add_argument("--dashboard-port", type=int, default=8080)

    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Run: pip install wandb")
        sys.exit(1)

    api = wandb.Api()

    # List artifacts
    if args.list:
        print(f"\n=== Available Artifacts in {args.entity}/{args.project} ===\n")
        try:
            # Get artifacts of type optuna-study
            artifacts = api.artifacts(
                type_name="optuna-study",
                name=f"{args.entity}/{args.project}",
            )
            for artifact in artifacts:
                print(f"  {artifact.name}:{artifact.version}")
                print(f"    Created: {artifact.created_at}")
                print(f"    Size: {artifact.size / (1024*1024):.2f} MB")
                print()
        except Exception as e:
            print(f"Error listing artifacts: {e}")
            print("\nTrying alternative projects...")
            for proj in ["sella-hpo", "hip-multi-mode-hpo", "scine-multi-mode-hpo", "hpo-databases"]:
                try:
                    arts = list(api.artifacts(type_name="optuna-study", name=f"{args.entity}/{proj}"))
                    if arts:
                        print(f"\n  Project: {proj}")
                        for a in arts:
                            print(f"    - {a.name}:{a.version}")
                except:
                    pass
        return

    # Download artifact(s)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_dbs = []

    if args.name:
        artifact_path = f"{args.entity}/{args.project}/{args.name}:latest"
        print(f"\nDownloading: {artifact_path}")
        try:
            artifact = api.artifact(artifact_path)
            download_path = artifact.download(root=str(output_dir))
            print(f"Downloaded to: {download_path}")

            # Find the .db file
            db_files = list(Path(download_path).glob("*.db"))
            downloaded_dbs.extend(db_files)
        except Exception as e:
            print(f"Error downloading {args.name}: {e}")
            print("Try: python scripts/download_hpo_artifacts.py --list")

    elif args.all:
        print(f"\nDownloading all artifacts to {output_dir}")
        for proj in ["sella-hpo", "hip-multi-mode-hpo", "scine-multi-mode-hpo", "hpo-databases"]:
            try:
                artifacts = list(api.artifacts(type_name="optuna-study", name=f"{args.entity}/{proj}"))
                for artifact in artifacts:
                    print(f"  Downloading: {artifact.name}")
                    try:
                        download_path = artifact.download(root=str(output_dir / proj))
                        db_files = list(Path(download_path).glob("*.db"))
                        downloaded_dbs.extend(db_files)
                    except Exception as e:
                        print(f"    Error: {e}")
            except:
                pass

    # Launch Optuna dashboard if requested
    if args.dashboard and downloaded_dbs:
        db_path = downloaded_dbs[0]
        print(f"\n=== Launching Optuna Dashboard ===")
        print(f"Database: {db_path}")
        print(f"URL: http://localhost:{args.dashboard_port}")
        print("Press Ctrl+C to stop\n")

        try:
            subprocess.run([
                sys.executable, "-m", "optuna_dashboard",
                f"sqlite:///{db_path}",
                "--port", str(args.dashboard_port),
            ])
        except KeyboardInterrupt:
            print("\nDashboard stopped")
        except FileNotFoundError:
            print("ERROR: optuna-dashboard not installed. Run: pip install optuna-dashboard")

    if downloaded_dbs:
        print(f"\n=== Downloaded Databases ===")
        for db in downloaded_dbs:
            print(f"  {db}")
        print(f"\nTo analyze locally:")
        print(f"  python hpo_results/analyze_hpo_results.py")
        print(f"\nTo launch Optuna dashboard:")
        print(f"  optuna-dashboard sqlite:///{downloaded_dbs[0]}")


if __name__ == "__main__":
    main()
