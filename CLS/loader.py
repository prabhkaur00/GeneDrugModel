import os, io, json
from glob import glob
from typing import Dict
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

def _to_gene_key(v):
    # Normalize CSV values like "1234.0" → "1234"
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)

class ProteinDrugInteractionDataset(Dataset):
    """
    Loads protein embeddings from multiple LMDB shards using a precomputed
    gene_id -> lmdb_path mapping (geneid_to_lmdb.json). Avoids any DB scans.
    """
    def __init__(
        self,
        seg_df,
        protein_lmdb_dir: str,
        smiles_cache: Dict[str, "torch_geometric.data.Data"],
        gid2lmdb_json: str,
        verify_keys: bool = False,
        verbose: bool = True,
    ):
        self.df = seg_df.reset_index(drop=True)
        self.smiles_cache = smiles_cache

        # Load mapping and normalize to absolute paths under protein_lmdb_dir
        with open(gid2lmdb_json) as f:
            raw_map = json.load(f)

        # Some mappings might be absolute already; normalize relative to dir if needed
        def _normalize_path(p):
            return p if os.path.isabs(p) else os.path.abspath(os.path.join(protein_lmdb_dir, p))

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
            # subdir=False because your shards are *.lmdb files (not directories)
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
            # Should not happen if paths were filtered; open on the fly as a fallback
            env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False, readahead=True, max_readers=512)
            self.env_by_path[lmdb_path] = env

        key = gid.encode("utf-8")
        with env.begin(buffers=True) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"gene_id {gid} not found in LMDB {lmdb_path}")
            protein_embedding = np.load(io.BytesIO(bytes(buf)), allow_pickle=False)
            protein_embedding = torch.tensor(protein_embedding, dtype=torch.float32)

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