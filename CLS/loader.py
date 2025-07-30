import torch
from torch.utils.data import Dataset
import numpy as np
import lmdb
import io
import torch
from torch.utils.data import Dataset
import lmdb
import io
import numpy as np

class ProteinDrugInteractionDataset(Dataset):
    def __init__(self, seg_df, protein_lmdb_path, smiles_cache):
        self.df = seg_df.reset_index(drop=True)
        self.lmdb_path = protein_lmdb_path
        self.smiles_cache = smiles_cache
        self._env = None

    def _lazy_env(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=False)
        return self._env

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gene_id = str(int(float(row["gene_id"])))  # Ensure clean integer string
        smiles = row["smiles"]

        with self._lazy_env().begin() as txn:
            emb_bytes = txn.get(gene_id.encode("utf-8"))
            if emb_bytes is None:
                raise KeyError(f"GeneID {gene_id} not found in LMDB.")
            protein_embedding = np.load(io.BytesIO(emb_bytes))
            protein_embedding = torch.tensor(protein_embedding, dtype=torch.float32)

        if smiles not in self.smiles_cache:
            raise KeyError(f"SMILES {smiles} not found in smiles_cache.")
        drug_embedding = self.smiles_cache[smiles]

        return {
            "protein": protein_embedding,
            "drug": drug_embedding,
            "target_id": int(row["target_id"]),
            "direction_id": int(row["direction_id"]),
            "row_idx": int(row["row_idx"]),
            "seg_idx": int(row["seg_idx"])
        }

    def __len__(self):
        return len(self.df)
    
def collate_fn(batch):
    protein_batch = torch.stack([b["protein"] for b in batch])
    drug_batch    = torch.stack([b["drug"] for b in batch])
    target_ids    = torch.tensor([b["target_id"] for b in batch], dtype=torch.long)
    direction_ids = torch.tensor([b["direction_id"] for b in batch], dtype=torch.long)
    row_idxs      = [b["row_idx"] for b in batch]
    seg_idxs      = [b["seg_idx"] for b in batch]

    return protein_batch, drug_batch, target_ids, direction_ids, row_idxs, seg_idxs