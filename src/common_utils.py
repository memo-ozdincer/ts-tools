# common_utils.py
import os
import argparse
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as TGDData
from torch_geometric.loader import DataLoader

# These were common imports in both scripts
from transition1x import Dataloader as T1xDataloader
from hip.equiformer_torch_calculator import EquiformerTorchCalculator

# --- Shared Dataset Class ---
# src/common_utils.py

# ... (imports and other functions remain the same) ...

# --- MODIFIED Shared Dataset Class ---
class Transition1xDataset(Dataset):
    """Loads transition state data from the Transition1x HDF5 file."""
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
                # --- NEW: Extract reactant data ---
                reactant = mol["reactant"]

                # Ensure reactant and TS have the same atom ordering/count
                if len(ts["atomic_numbers"]) != len(reactant["atomic_numbers"]):
                    print(f"[WARN] Skipping idx={idx} due to atom count mismatch between reactant and TS.")
                    continue
                
                data = TGDData(
                    z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                    pos_transition=torch.tensor(ts["positions"], dtype=torch.float),
                    # --- NEW: Store reactant positions in the data object ---
                    pos_reactant=torch.tensor(reactant["positions"], dtype=torch.float),
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

# --- Shared Transform Class ---
class UsePos:
    """A pre-transform to set data.pos from a specified attribute."""
    def __init__(self, attr: str = "pos_transition"):
        self.attr = attr

    def __call__(self, data: TGDData) -> TGDData:
        pos = getattr(data, self.attr, None)
        if pos is None:
            raise ValueError(f"Data missing '{self.attr}'. Keys: {list(data.keys())}")
        data.pos = pos
        return data

# --- Shared Command-line Argument Helper ---
def add_common_args(parser: argparse.ArgumentParser):
    """Adds common arguments for data, model, and output paths."""
    PROJECT = "/project/memo"
    parser.add_argument("--max-samples", type=int, default=30, help="Max samples to process.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., 'test', 'validation').")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--h5-path", type=str, default=os.path.join(PROJECT, "large-files", "data", "transition1x.h5"))
    parser.add_argument("--checkpoint-path", type=str, default=os.path.join(PROJECT, "large-files", "ckpt", "hip_v2.ckpt"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT, "large-files", "graphs", "out"))
    return parser

# --- Noise Generation Helper ---
def add_gaussian_noise_to_coords(coords: torch.Tensor, noise_rms_angstrom: float) -> torch.Tensor:
    """
    Add Gaussian noise to coordinates with specified RMS amplitude.

    Args:
        coords: (N, 3) or (3N,) tensor of coordinates
        noise_rms_angstrom: Target RMS displacement in Angstroms

    Returns:
        Noisy coordinates with same shape as input
    """
    original_shape = coords.shape
    coords_flat = coords.reshape(-1, 3)

    # Generate Gaussian noise with std = noise_rms_angstrom
    # This gives RMS displacement ≈ noise_rms_angstrom
    noise = torch.randn_like(coords_flat) * noise_rms_angstrom

    noisy_coords = coords_flat + noise
    return noisy_coords.reshape(original_shape)


def parse_starting_geometry(start_from: str, batch, noise_seed: Optional[int] = None):
    """
    Parse starting geometry specification and apply noise if requested.

    Supports formats like:
    - "reactant", "ts", "midpoint_rt", "three_quarter_rt" (standard geometries)
    - "reactant_noise0.5A", "reactant_noise1A", etc. (noisy geometries)

    Args:
        start_from: String specifying the starting geometry
        batch: Data batch containing pos_reactant, pos_transition, etc.
        noise_seed: Optional random seed for reproducible noise

    Returns:
        torch.Tensor: Starting coordinates
    """
    # Check if noise is requested
    if "_noise" in start_from:
        # Parse format: "reactant_noise1A" or "reactant_noise0.5A"
        parts = start_from.split("_noise")
        base_geom = parts[0]
        noise_str = parts[1].rstrip("A")  # Remove trailing 'A'
        try:
            noise_level = float(noise_str)
        except ValueError:
            raise ValueError(f"Invalid noise level in '{start_from}'. Expected format: 'reactant_noise1A'")

        # Get base geometry
        if base_geom == "reactant":
            initial_coords = batch.pos_reactant
        elif base_geom == "ts":
            initial_coords = batch.pos_transition
        elif base_geom == "midpoint_rt":
            initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
        elif base_geom == "three_quarter_rt":
            initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
        else:
            raise ValueError(f"Unknown base geometry '{base_geom}' in '{start_from}'")

        # Set random seed if provided for reproducibility
        if noise_seed is not None:
            torch.manual_seed(noise_seed)

        # Add noise
        initial_coords = add_gaussian_noise_to_coords(initial_coords.clone(), noise_level)

    else:
        # Standard geometries (no noise)
        if start_from == "reactant":
            initial_coords = batch.pos_reactant
        elif start_from == "ts":
            initial_coords = batch.pos_transition
        elif start_from == "midpoint_rt":
            initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
        elif start_from == "three_quarter_rt":
            initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
        else:
            raise ValueError(f"Unknown starting geometry: {start_from}")

    return initial_coords


# --- Shared Experiment Setup Function ---
def setup_experiment(args: argparse.Namespace, batch_size: int = 1, shuffle: bool = False, dataset_load_multiplier: int = 1) -> Tuple[EquiformerTorchCalculator, DataLoader, str, str]:
    """
    Handles all boilerplate setup: model loading, dataset creation, etc.

    Returns:
        A tuple of (calculator, dataloader, device, output_directory).
    """
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint_path}")
    calculator = EquiformerTorchCalculator(
        checkpoint_path=args.checkpoint_path,
        hessian_method="predict",
    )
    
    # Prepare data
    print(f"Loading dataset: {args.h5_path} (split={args.split})")
    dataset = Transition1xDataset(
        h5_path=args.h5_path,
        split=args.split,
        max_samples=args.max_samples * dataset_load_multiplier,
        transform=UsePos("pos_transition"),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")

    print(f"Device:     {args.device}")
    print(f"Loaded {len(dataset)} candidate samples.")
    print("-" * 30)

    return calculator, dataloader, args.device, args.out_dir