# gad_frequency_analysis.py
import os
import json
import argparse
from typing import Dict, Any, List

import torch

# Import shared utilities
from .common_utils import setup_experiment, add_common_args
# Import the required analysis function
from hip.frequency_analysis import analyze_frequencies_torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run frequency analysis on Transition1x samples.")
    parser = add_common_args(parser)
    args = parser.parse_args()

    # Use the setup function to get model, data, and config
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    results_summary: List[Dict[str, Any]] = []

    print(f"Analyzing up to {args.max_samples} samples for vibrational frequencies")

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break
        try:
            batch.natoms = torch.tensor([batch.pos.shape[0]], dtype=torch.long)
            batch = batch.to(device)

            results = calculator.predict(batch, do_hessian=True)
            hess = results["hessian"]
            pos = batch.pos
            atomic_nums = batch.z

            # analyze_frequencies_torch does Eckart projection + eigendecomp
            freq_info = analyze_frequencies_torch(hess, pos, atomic_nums)

            out = {
                "index": i,
                "natoms": int(pos.shape[0]),
                "neg_num": int(freq_info["neg_num"]),
                "eigvals": freq_info["eigvals"].detach().cpu().numpy().tolist(),
            }
            if freq_info.get("eigvecs") is not None:
                out["eigvecs"] = freq_info["eigvecs"].detach().cpu().numpy().tolist()

            results_summary.append(out)
            print(f"[{i}] N={out['natoms']}, neg_num={out['neg_num']}")
        except Exception as e:
            print(f"[{i}] ERROR: {e}")

    out_json = os.path.join(out_dir, f"rgd1_frequency_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved frequency analysis summary for {len(results_summary)} samples → {out_json}")