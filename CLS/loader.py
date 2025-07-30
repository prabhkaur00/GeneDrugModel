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
from torch_geometric.data import Batch

class ProteinDrugInteractionDataset(Dataset):
    def __init__(self, seg_df, protein_lmdb_path, smiles_cache):
        self.df = seg_df.reset_index(drop=True)
        self.lmdb_path = protein_lmdb_path
        self.smiles_cache = smiles_cache
        self._env = None
        self.valid_indices = []
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            self.valid_keys = {k.decode("utf-8") for k, _ in txn.cursor()}
        env.close()
        for idx, row in self.df.iterrows():
            gene_id = str(int(float(row["gene_id"])))
            smiles = row["smiles"]

            if gene_id in self.valid_keys and smiles in self.smiles_cache:
                self.valid_indices.append(idx)

        print(f"[Dataset] Loaded {len(self.valid_indices)} valid entries out of {len(self.df)}")

    def _lazy_env(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=False)
        return self._env

    def __getitem__(self, idx):
        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]
        gene_id = str(int(float(row["gene_id"])))
        smiles = row["smiles"]

        with self._lazy_env().begin() as txn:
            emb_bytes = txn.get(gene_id.encode("utf-8"))
            protein_embedding = np.load(io.BytesIO(emb_bytes))
            protein_embedding = torch.tensor(protein_embedding, dtype=torch.float32)

        drug_embedding = self.smiles_cache[smiles]

        return {
            "protein": protein_embedding,
            "drug": drug_embedding,  # <- this should be a PyG `Data` object, if model expects a graph
            "target_id": int(row["target_id"]),
            "direction_id": int(row["direction_id"]),
            "row_idx": int(row["row_idx"]),
            "seg_idx": int(row["seg_idx"])
        }

    def __len__(self):
        return len(self.valid_indices)


   
def collate_fn(batch):
    """Collate function for protein-drug interaction dataset using GNN and classification heads."""
    
    # Filter out None entries
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise ValueError("All items in batch are None")

    # Stack fixed-size protein embeddings
    protein_embeddings = torch.stack([item['protein'] for item in batch])  # (B, 768)

    # Batch PyG graphs
    drug_graphs = [item['drug'] for item in batch]
    batched_drugs = Batch.from_data_list(drug_graphs)  # PyG Batch object

    # Classification labels
    target_ids    = torch.tensor([item['target_id'] for item in batch], dtype=torch.long)
    direction_ids = torch.tensor([item['direction_id'] for item in batch], dtype=torch.long)

    # Optional row metadata
    row_idxs = [item['row_idx'] for item in batch]
    seg_idxs = [item['seg_idx'] for item in batch]

    return {
        "protein_embeddings": protein_embeddings,
        "drug_graphs": batched_drugs,
        "target_ids": target_ids,
        "direction_ids": direction_ids,
        "row_idxs": row_idxs,
        "seg_idxs": seg_idxs}