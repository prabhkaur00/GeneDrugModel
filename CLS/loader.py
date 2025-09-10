import os, io, json
from glob import glob
from typing import Dict, List, Tuple, Any
import lmdb
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Batch as PyGBatch

def _to_gene_key(v):
    # Normalize CSV values like "1234.0" → "1234"
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)

def _pad_stack(seq_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Pad a list of [Ti, dim] DNA embedding tensors into:
      - batch: [B, Tmax, dim] (zero-padded)
      - mask : [B, Tmax] (True = PAD)
    """
    if len(seq_list) == 0:
        raise ValueError("Empty sequence list passed to _pad_stack.")
    B = len(seq_list)
    dims = [x.dim() for x in seq_list]
    if not all(d in (1, 2) for d in dims):
        raise ValueError("All DNA embeddings must be rank-1 ([dim]) or rank-2 ([T, dim]).")

    # Coerce any [dim] vectors to [1, dim] so we can treat them as 1-token sequences
    normed = [x.unsqueeze(0) if x.dim() == 1 else x for x in seq_list]  # each: [Ti, dim]
    T = max(x.size(0) for x in normed)
    d = normed[0].size(1)

    batch = normed[0].new_zeros((B, T, d))           # zero-pad
    mask  = torch.ones(B, T, dtype=torch.bool)       # True=PAD
    for i, x in enumerate(normed):
        t = x.size(0)
        batch[i, :t] = x
        mask[i, :t]  = False
    return batch, mask

class ProteinDrugInteractionDataset(Dataset):
    """
    Loads DNA (formerly 'protein') embeddings from multiple LMDB shards using a precomputed
    gene_id -> lmdb_path mapping (geneid_to_lmdb.json). Avoids any DB scans.

    Returns per-item:
        {
          "dna": Tensor [T, dim] or [dim] (auto-handled in collate),
          "drug": PyG Data,
          "target_id": int,         # kept for compatibility (unused in direction-only runs)
          "direction_id": int,      # label for direction (binary)
          "row_idx": int,
          "seg_idx": int,
        }
    """
    def __init__(
        self,
        seg_df,
        dna_lmdb_dir: str,                           # renamed from protein_lmdb_dir
        smiles_cache: Dict[str, "torch_geometric.data.Data"],
        gid2lmdb_json: str,
        verify_keys: bool = False,
        verbose: bool = True,
    ):
        self.df = seg_df.reset_index(drop=True)
        self.smiles_cache = smiles_cache

        # Load mapping and normalize to absolute paths under dna_lmdb_dir
        with open(gid2lmdb_json) as f:
            raw_map = json.load(f)

        def _normalize_path(p):
            return p if os.path.isabs(p) else os.path.abspath(os.path.join(dna_lmdb_dir, p))

        self.gid2path = { _to_gene_key(g): _normalize_path(p) for g, p in raw_map.items() }

        # Keep only gene_ids that appear in this split (reduces opened envs)
        genes_needed = {_to_gene_key(g) for g in self.df["gene_id"].tolist()}
        genes_needed &= set(self.gid2path.keys())

        # Build set of LMDB shard paths that are actually needed
        self.paths_needed = sorted({ self.gid2path[g] for g in genes_needed })

        if verbose:
            print(f"[Dataset] genes in split: {len(set(self.df['gene_id']))} | "
                  f"genes with mapping: {len(genes_needed)} | "
                  f"lmdb shards needed: {len(self.paths_needed)}")

        # Open only required LMDB envs (readonly, no locks)
        self.env_by_path = {}
        for p in self.paths_needed:
            # subdir=False because shards are *.lmdb files (not directories)
            env = lmdb.open(p, readonly=True, lock=False, subdir=False, readahead=True, max_readers=512)
            self.env_by_path[p] = env

        # Precompute valid indices (smiles present + mapped gene)
        self.valid_indices = []
        missing_gene_map = 0
        missing_smiles = 0
        missing_key_in_env = 0

        if verify_keys:
            # Optional: confirm key presence in LMDB (costly but deterministic)
            for idx, row in self.df.iterrows():
                gid = _to_gene_key(row["gene_id"])
                smi = row["smiles"]
                pth = self.gid2path.get(gid, None)
                if pth is None:
                    missing_gene_map += 1
                    continue
                if smi not in self.smiles_cache:
                    missing_smiles += 1
                    continue
                key = gid.encode("utf-8")
                with self.env_by_path[pth].begin(buffers=True) as txn:
                    if txn.get(key) is None:
                        missing_key_in_env += 1
                        continue
                self.valid_indices.append(idx)
        else:
            for idx, row in self.df.iterrows():
                gid = _to_gene_key(row["gene_id"])
                smi = row["smiles"]
                if (gid in self.gid2path) and (smi in self.smiles_cache):
                    self.valid_indices.append(idx)
                else:
                    missing_gene_map += int(gid not in self.gid2path)
                    missing_smiles   += int(smi not in self.smiles_cache)

        if verbose:
            print(f"[Dataset] valid={len(self.valid_indices)} / {len(self.df)} "
                  f"(missing_map={missing_gene_map}, missing_smiles={missing_smiles}, "
                  f"missing_key_in_env={missing_key_in_env})", flush=True)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        df_idx = self.valid_indices[i]
        row = self.df.iloc[df_idx]

        gid = _to_gene_key(row["gene_id"])
        smi = row["smiles"]

        lmdb_path = self.gid2path.get(gid, None)
        if lmdb_path is None:
            raise KeyError(f"gene_id {gid} has no LMDB mapping")

        env = self.env_by_path.get(lmdb_path, None)
        if env is None:
            env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False, readahead=True, max_readers=512)
            self.env_by_path[lmdb_path] = env

        key = gid.encode("utf-8")
        with env.begin(buffers=True) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"gene_id {gid} not found in LMDB {lmdb_path}")
            dna_embedding = np.load(io.BytesIO(bytes(buf)), allow_pickle=False)
            dna_embedding = torch.tensor(dna_embedding, dtype=torch.float32)
            # Supports either [dim] (legacy) or [T, dim] (token sequence)
            if dna_embedding.dim() == 1:
                # treat as 1-token sequence to keep the new model happy
                dna_embedding = dna_embedding.unsqueeze(0)

        drug_graph = self.smiles_cache[smi]

        return {
            "dna": dna_embedding,                     # [T, dim] (or [1, dim] if legacy)
            "drug": drug_graph,                       # PyG Data
            "target_id":    int(row.get("target_id", -1)),  # optional / unused in direction-only runs
            "direction_id": int(row["direction_id"]),
            "row_idx": int(row.get("row_idx", -1)),
            "seg_idx": int(row.get("seg_idx", 0)),
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("All items in batch are None")

    # --- DNA embeddings: list of [Ti, dim] → [B, T, dim] + mask [B, T] ---
    dna_list: List[Tensor] = [b["dna"] for b in batch]
    dna_embeddings, dna_mask = _pad_stack(dna_list)   # zero-padded

    # --- Drug graphs: to PyG Batch ---
    drug_graphs = PyGBatch.from_data_list([b["drug"] for b in batch])

    # --- Labels ---
    direction_ids = torch.tensor([b["direction_id"] for b in batch], dtype=torch.long)
    # Keep target_ids for compatibility if present (not used in direction-only run)
    if "target_id" in batch[0]:
        target_ids = torch.tensor([b.get("target_id", -1) for b in batch], dtype=torch.long)
    else:
        target_ids = torch.full((len(batch),), -1, dtype=torch.long)

    return {
        "dna_embeddings": dna_embeddings,   # [B, T, dim]
        "dna_mask": dna_mask,               # [B, T], True = PAD
        "drug_graphs": drug_graphs,         # PyG Batch
        "direction_ids": direction_ids,     # [B]
        "target_ids": target_ids,           # [B] (kept for API symmetry)
        "row_idxs": [b.get("row_idx", -1) for b in batch],
        "seg_idxs": [b.get("seg_idx", 0) for b in batch],
    }
