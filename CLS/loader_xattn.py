import os, io, json
from typing import Dict
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

def _to_gene_key(v):
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)

class ProteinDrugInteractionDataset(Dataset):
    def __init__(
        self,
        seg_df,
        lmdb_dir: str,
        geneid_to_lmdb_json: str,
        smiles_cache: Dict[str, "torch_geometric.data.Data"],
        verify_keys: bool = False,
        verbose: bool = True,
    ):
        self.df = seg_df.reset_index(drop=True)
        self.smiles_cache = smiles_cache
        self.lmdb_dir = os.path.abspath(lmdb_dir)
        if not os.path.isdir(self.lmdb_dir):
            raise NotADirectoryError(self.lmdb_dir)
        with open(geneid_to_lmdb_json, "r") as f:
            raw = json.load(f)

        self.geneid_to_shard = {}
        for k, v in raw.items():
            try:
                shard = int(v)
            except (ValueError, TypeError):
                # v is likely a path like ".../full_embeddings_0.h5.lmdb"
                shard = int(os.path.basename(str(v)).split('_')[-1].split('.')[0])
            self.geneid_to_shard[_to_gene_key(k)] = shard

        self.envs: Dict[int, lmdb.Environment] = {}
        self.valid_indices = []
        missing_smiles = 0
        missing_map = 0
        missing_key_in_env = 0

        if verify_keys:
            for idx, row in self.df.iterrows():
                gid = _to_gene_key(row["gene_id"])
                smi = row["smiles"]
                if smi not in self.smiles_cache:
                    missing_smiles += 1
                    continue
                shard = self.geneid_to_shard.get(gid, None)
                if shard is None:
                    missing_map += 1
                    continue
                env = self._open_env(shard)
                # CHANGED: use buffers=False to avoid txn-backed memoryviews
                with env.begin(buffers=False) as txn:
                    if txn.get(gid.encode("utf-8")) is None:
                        missing_key_in_env += 1
                        continue
                self.valid_indices.append(idx)
        else:
            for idx, row in self.df.iterrows():
                gid = _to_gene_key(row["gene_id"])
                smi = row["smiles"]
                if smi not in self.smiles_cache:
                    missing_smiles += 1
                    continue
                if gid not in self.geneid_to_shard:
                    missing_map += 1
                    continue
                self.valid_indices.append(idx)

        if verbose:
            print(f"[Dataset] LMDB dir: {self.lmdb_dir}")
            print(f"[Dataset] shards opened lazily; mapping size={len(self.geneid_to_shard)}")
            print(f"[Dataset] valid={len(self.valid_indices)} / {len(self.df)} (missing_smiles={missing_smiles}, missing_map={missing_map}, missing_key_in_env={missing_key_in_env})", flush=True)

    def _open_env(self, shard_idx: int) -> lmdb.Environment:
        if shard_idx in self.envs:
            return self.envs[shard_idx]
        path = os.path.join(self.lmdb_dir, f"full_embeddings_{shard_idx}.h5.lmdb")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        env = lmdb.open(path, readonly=True, lock=False, subdir=False, readahead=True, max_readers=2048)
        self.envs[shard_idx] = env
        return env

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        df_idx = self.valid_indices[i]
        row = self.df.iloc[df_idx]
        gid = _to_gene_key(row["gene_id"])
        smi = row["smiles"]
        shard = self.geneid_to_shard.get(gid, None)
        if shard is None:
            raise KeyError(f"gene_id {gid} missing in mapping")
        env = self._open_env(shard)
        # CHANGED: buffers=False so txn.get returns plain bytes (pickle-safe)
        with env.begin(buffers=False) as txn:
            buf = txn.get(gid.encode("utf-8"))
            if buf is None:
                raise KeyError(f"gene_id {gid} not found in shard {shard}")
            emb = np.load(io.BytesIO(buf), allow_pickle=False)  # CHANGED: buf already bytes
            protein_embedding = torch.tensor(emb, dtype=torch.float32)
        drug_graph = self.smiles_cache[smi]
        return {
            "protein": protein_embedding,
            "drug": drug_graph,
            "target_id": int(row["target_id"]),
            "direction_id": int(row["direction_id"]),
            "row_idx": int(row.get("row_idx", -1)),
            "seg_idx": int(row.get("seg_idx", 0)),
        }
    
    def __getstate__(self):
    # Remove unpicklable LMDB environments before pickling
        state = self.__dict__.copy()
        state['envs'] = {}  # Clear the environments dict
        return state

    def __setstate__(self, state):
        # Restore state and let environments be opened lazily
        self.__dict__.update(state)
        self.envs = {}


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("All items in batch are None")
    protein_embeddings = pad_sequence([b["protein"] for b in batch], batch_first=True)  # (B, T_max, 768)

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
