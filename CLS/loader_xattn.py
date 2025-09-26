import os, io
import pickle
import lmdb
import numpy as np
import torch
from typing import Dict
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
        lmdb_path: str,
        smiles_cache: Dict[str, "torch_geometric.data.Data"],
        verify_keys: bool = False,
        verbose: bool = True,
    ):
        self.df = seg_df.reset_index(drop=True)
        self.smiles_cache = smiles_cache
        self.lmdb_path = os.path.abspath(lmdb_path)
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(self.lmdb_path)

        # Open the filtered LMDB once
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            subdir=os.path.isdir(self.lmdb_path),
            max_readers=2048,
            max_dbs=1
        )

        self.valid_indices = []
        missing_smiles = 0
        missing_key_in_env = 0

        if verify_keys:
            with self.env.begin(buffers=False) as txn:
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
            for idx, row in self.df.iterrows():
                gid = _to_gene_key(row["gene_id"])
                smi = row["smiles"]
                if smi not in self.smiles_cache:
                    missing_smiles += 1
                    continue
                self.valid_indices.append(idx)

        if verbose:
            print(f"[Dataset] Filtered LMDB path: {self.lmdb_path}")
            print(f"[Dataset] valid={len(self.valid_indices)} / {len(self.df)} "
                  f"(missing_smiles={missing_smiles}, missing_key_in_env={missing_key_in_env})", flush=True)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        df_idx = self.valid_indices[i]
        row = self.df.iloc[df_idx]
        gid = _to_gene_key(row["gene_id"])
        smi = row["smiles"]

        with self.env.begin(buffers=False) as txn:
            buf = txn.get(gid.encode("utf-8"))
            if buf is None:
                raise KeyError(f"gene_id {gid} not found in LMDB")
            emb = np.load(io.BytesIO(buf), allow_pickle=False)
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
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            subdir=os.path.isdir(self.lmdb_path),
            max_readers=2048,
            max_dbs=1
        )

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("All items in batch are None")

    prots = [b["protein"] for b in batch]                       # list of [Ti, 768]
    lengths_p = torch.tensor([t.shape[0] for t in prots])
    protein_embeddings = pad_sequence(prots, batch_first=True)   # [B, Tp_max, 768]
    Tp_max = protein_embeddings.size(1)
    ar = torch.arange(Tp_max)
    protein_pad_mask = ar.unsqueeze(0) >= lengths_p.unsqueeze(1) # [B, Tp_max], bool

    drug_graphs = Batch.from_data_list([b["drug"] for b in batch])

    target_ids    = torch.tensor([b["target_id"] for b in batch], dtype=torch.long)
    direction_ids = torch.tensor([b["direction_id"] for b in batch], dtype=torch.long)

    return {
        "protein_embeddings": protein_embeddings,   # float32
        "protein_pad_mask": protein_pad_mask,       # bool
        "drug_graphs": drug_graphs,                 # PyG Batch
        "target_ids": target_ids,
        "direction_ids": direction_ids,
        "row_idxs": [b["row_idx"] for b in batch],
        "seg_idxs": [b["seg_idx"] for b in batch],
    }
