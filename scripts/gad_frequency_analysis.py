import os
import json
from typing import Dict, Any, List

import torch
import numpy as np
from torch_geometric.loader import DataLoader

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.ff_lmdb import LmdbDataset
from ocpmodels.hessian_graph_transform import HessianGraphTransform

# --- frequency analysis utilities ---
from hip.frequency_analysis import analyze_frequencies_torch  # Eckart projection + eigendecomp


# -----------------------------
# Pre-transform: use TS coords
# -----------------------------
class UsePos:
    def __init__(self, attr: str = "pos_transition"):
        self.attr = attr

    def __call__(self, data):
        pos = getattr(data, self.attr, None)
        if pos is None:
            raise ValueError(f"Data missing '{self.attr}'. Keys: {list(data.keys())}")
        data.pos = pos
        return data


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = os.path.expanduser("~")
    PROJECT = "/project/memo"  # CSLab project space (large, persistent)

    # where big files live
    checkpoint_path = os.path.join(PROJECT, "/project/memo/large-files/ckpt/hesspred_v1.ckpt")
    dataset_path = os.path.join(PROJECT, "/project/memo/large-files/Transition1x/data/t1x_lmdb/t1x_test_ts.lmdb")

    # where to write results (small)
    out_dir = os.path.join(HOME, "large-files", "out")
    os.makedirs(out_dir, exist_ok=True)

    # load model
    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )

    # transforms (use TS coordinates; build hessian graph)
    use_pos_tf = UsePos("pos_transition")
    hess_tf = HessianGraphTransform(
        cutoff=calculator.potential.cutoff,
        max_neighbors=calculator.potential.max_neighbors,
        use_pbc=getattr(calculator.potential, "use_pbc", False),
    )
    composed_tf = lambda d: hess_tf(use_pos_tf(d))

    # data
    dataset = LmdbDataset(dataset_path, transform=composed_tf)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    MAX_SAMPLES = 30
    results_summary: List[Dict[str, Any]] = []

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {dataset_path}")
    print(f"Device:     {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Analyzing up to {MAX_SAMPLES} samples for vibrational frequencies")

    for i, batch in enumerate(dataloader):
        if i >= MAX_SAMPLES:
            break
        try:
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
            if freq_info.get("eigvecs", None) is not None:
                out["eigvecs"] = freq_info["eigvecs"].detach().cpu().numpy().tolist()

            results_summary.append(out)
            print(f"[{i}] N={out['natoms']}, neg_num={out['neg_num']}")
        except Exception as e:
            print(f"[{i}] ERROR: {e}")

    out_json = os.path.join(out_dir, f"rgd1_frequency_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved frequency analysis summary for {len(results_summary)} samples → {out_json}")