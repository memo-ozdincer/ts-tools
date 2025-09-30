# Just put this one in the shared directory
import os
import lmdb
import pickle
import torch
from torch_geometric.data import Data as TGDData
from tqdm import tqdm

from transition1x import Dataloader  # pip install .[example]


def convert_t1x_to_lmdb(h5_path: str, out_dir: str, split: str = "train", max_samples: int = None):
    """
    Args:
        h5_path (str): Path to Transition1x .h5 file
        out_dir (str): Where to save LMDBs
        split (str): 'train', 'val', 'test'
        max_samples (int, optional): Cap number of samples
    """
    os.makedirs(out_dir, exist_ok=True)
    lmdb_path = os.path.join(out_dir, f"t1x_{split}_ts.lmdb")

    # remove existing
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    if os.path.exists(lmdb_path + "-lock"):
        os.remove(lmdb_path + "-lock")

    dataloader = Dataloader(h5_path, datasplit=split, only_final=True)

    map_size = 10 * 1024 * 1024 * 1024  # 10GB
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False)

    n_written = 0
    with env.begin(write=True) as txn:
        for idx, mol in tqdm(enumerate(dataloader), desc=f"Writing {split} split"):
            if max_samples is not None and n_written >= max_samples:
                break
            try:
                ts = mol["transition_state"]

                data = TGDData(
                    z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                    pos_transition=torch.tensor(ts["positions"], dtype=torch.float),
                    energy=torch.tensor(ts["wB97x_6-31G(d).energy"], dtype=torch.float),
                    forces=torch.tensor(ts["wB97x_6-31G(d).forces"], dtype=torch.float),
                    rxn=mol["rxn"],
                    formula=mol["formula"],
                )

                txn.put(
                    f"{idx}".encode("ascii"),
                    pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
                )
                n_written += 1
            except Exception as e:
                print(f"[WARN] Skipping idx={idx} due to error: {e}")
                continue

        txn.put("length".encode("ascii"), pickle.dumps(n_written))
    env.close()

    print(f"✔ Wrote {n_written} TS samples → {lmdb_path}")


if __name__ == "__main__":
    h5_file = "/project/memo/large-files/Transition1x/data/transition1x.h5"
    out_dir = "/project/memo/large-files/Transition1x/data/t1x_lmdb"

    # small debug split (first 100 samples)
    convert_t1x_to_lmdb(h5_file, out_dir, "train", max_samples=100)

    # full conversion (comment in when ready)
    # for split in ["train", "val", "test"]:
    #     convert_t1x_to_lmdb(h5_file, out_dir, split)