import os
import json
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as TGDData
from torch_geometric.loader import DataLoader

from transition1x import Dataloader as T1xDataloader

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from ocpmodels.hessian_graph_transform import HessianGraphTransform

# --- frequency analysis utilities ---
from hip.frequency_analysis import analyze_frequencies_torch  # Eckart projection + eigendecomp


class Transition1xDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.transform = transform
        self.samples: List[TGDData] = []

        loader = T1xDataloader(h5_path, datasplit=split, only_final=True)

        for idx, mol in enumerate(loader):
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            try:
                ts = mol["transition_state"]

                data = TGDData(
                    z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                    pos_transition=torch.tensor(ts["positions"], dtype=torch.float),
                    energy=torch.tensor(ts["wB97x_6-31G(d).energy"], dtype=torch.float),
                    forces=torch.tensor(ts["wB97x_6-31G(d).forces"], dtype=torch.float),
                    rxn=ts["rxn"],
                    formula=ts["formula"],
                )

                self.samples.append(data)
            except Exception as e:
                print(f"[WARN] Skipping idx={idx} due to error: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TGDData:
        data = self.samples[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


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
    PROJECT = "/project/memo" 

    MAX_SAMPLES = 30
    T1X_SPLIT = "test"

    # where big files live
    checkpoint_path = os.path.join(PROJECT, "large-files", "ckpt", "hesspred_v1.ckpt")
    h5_path = os.path.join(
        PROJECT,
        "large-files",
        "data",
        "transition1x.h5",
    )

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
    # hess_tf = HessianGraphTransform(
    #     cutoff=calculator.potential.cutoff,
    #     max_neighbors=calculator.potential.max_neighbors,
    #     use_pbc=getattr(calculator.potential, "use_pbc", False),
    # )

    # data
    dataset = Transition1xDataset(
        h5_path=h5_path,
        split=T1X_SPLIT,
        max_samples=MAX_SAMPLES,
        transform=use_pos_tf,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results_summary: List[Dict[str, Any]] = []

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {h5_path} (split={T1X_SPLIT})")
    print(f"Device:     {device}")
    print(f"Loaded samples: {len(dataset)}")
    print(f"Analyzing up to {MAX_SAMPLES} samples for vibrational frequencies")

    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")

    for i, batch in enumerate(dataloader):
        if i >= MAX_SAMPLES:
            break
        try:
            batch.natoms=torch.tensor([batch.pos.shape[1]], dtype=torch.long)
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
