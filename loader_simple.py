import os, io, json
from typing import Dict
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

def _to_gene_key(v):
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)

class ProteinDrugInteractionDataset(Dataset):
    """
    Reads mean-pooled gene embeddings directly from a single LMDB file.
    Keys are stringified gene_ids (e.g., "1234").
    """
    def __init__(
        self,
        seg_df,
        protein_lmdb_path: str,                   # <-- pass the .lmdb file here (e.g., ".../mean-pooled.all.lmdb")
        smiles_cache: Dict[str, "torch_geometric.data.Data"],
        verify_keys: bool = False,
        verbose: bool = True,
    ):
        self.df = seg_df.reset_index(drop=True)
        self.smiles_cache = smiles_cache
        self.lmdb_path = os.path.abspath(protein_lmdb_path)

        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB not found: {self.lmdb_path}")

        # Single LMDB env for all lookups
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            subdir=False,         # file, not directory
            readahead=True,
            max_readers=512
        )

        # Build valid indices (smiles present, gene has a key if verify_keys)
        self.valid_indices = []
        missing_smiles = 0
        missing_key_in_env = 0

        if verify_keys:
            with self.env.begin(buffers=True) as txn:
                for idx, row in self.df.iterrows():
                    gid = _to_gene_key(row["gene_id"])
                    smi = row["smiles"]
                    if smi not in self.smiles_cache:
                        missing_smiles += 1
                        continue
                    if txn.get(gid.encode("utf-8")) is None:
                        missing_key_in_env += 1
                        continue
                    self.valid_indices.append(idx)
        else:
            # Skip LMDB existence check for speed; fail fast in __getitem__ if missing
            for idx, row in self.df.iterrows():
                smi = row["smiles"]
                if smi in self.smiles_cache:
                    self.valid_indices.append(idx)
                else:
                    missing_smiles += 1

        if verbose:
            print(f"[Dataset] mean-pooled LMDB: {self.lmdb_path}")
            print(f"[Dataset] valid={len(self.valid_indices)} / {len(self.df)} "
                  f"(missing_smiles={missing_smiles}, missing_key_in_env={missing_key_in_env})", flush=True)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        df_idx = self.valid_indices[i]
        row = self.df.iloc[df_idx]

        gid = _to_gene_key(row["gene_id"])
        smi = row["smiles"]

        with self.env.begin(buffers=True) as txn:
            buf = txn.get(gid.encode("utf-8"))
            if buf is None:
                raise KeyError(f"gene_id {gid} not found in LMDB {self.lmdb_path}")
            emb = np.load(io.BytesIO(bytes(buf)), allow_pickle=False)  # shape (768,)
            protein_embedding = torch.tensor(emb, dtype=torch.float32)

        drug_graph = self.smiles_cache[smi]

        return {
            "protein": protein_embedding,                     # (768,)
            "drug":    drug_graph,                            # PyG Data
            "target_id":    int(row["target_id"]),
            "direction_id": int(row["direction_id"]),
            "row_idx": int(row.get("row_idx", -1)),
            "seg_idx": int(row.get("seg_idx", 0)),
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("All items in batch are None")
    protein_embeddings = torch.stack([b["protein"] for b in batch])  # (B, 768)
    drug_graphs = Batch.from_data_list([b["drug"] for b in batch])
    target_ids    = torch.tensor([b["target_id"] for b in batch], dtype=torch.long)
    direction_ids = torch.tensor([b["direction_id"] for b in batch], dtype=torch.long)
    return {
        "protein_embeddings": protein_embeddings,
        "drug_graphs": drug_graphs,
        "target_ids": target_ids,
        "direction_ids": direction_ids,
        "row_idxs": [b["row_idx"] for b in batch],
        "seg_idxs": [b["seg_idx"] for b in batch],
    }
