from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
import torch
from torch_geometric.data import Batch
from utils import cache_to_pyg_data
import re
import lmdb
import numpy as np
import io
drug_tag      = r"\[DRUG\]"
protein_tag   = r"\[PROTEIN[^\]]*\]"   # also catches “protein mutant form”, etc.
class ProteinDrugDataset(Dataset):
    def __init__(self, lmdb_path, csv_subset, graph_cache, tokenizer=None, max_len=25):
        self.data_df = csv_subset
        self.lmdb_path = lmdb_path
        self._env = None  # Will be lazily opened per worker
        self.smiles_cache = graph_cache
        self.tokenizer = tokenizer
        self.max_len = max_len
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            keys = {k.decode("utf-8") for k, _ in txn.cursor()}
        env.close()
        self.valid_indices = []
        for idx, row in self.data_df.iterrows():
            gene_id = str(int(row['GeneID']))
            smiles = row['SMILES']
            if gene_id in keys and smiles in self.smiles_cache:
                self.valid_indices.append(idx)

        print(f"Stage dataset loaded: {len(self.valid_indices)} valid entries out of {len(self.data_df)}")

    def _lazy_env(self):
        """Open LMDB env once per worker process."""
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=False)
        return self._env
    
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        df_idx = self.valid_indices[idx]
        row = self.data_df.iloc[df_idx]
        gene_id = str(int(row['GeneID']))
        smiles = row['SMILES']
        interaction = row['Interaction']

        interaction_phrase = self.extract_interaction_phrase(interaction)

        with self._lazy_env().begin() as txn:
            emb_bytes = txn.get(gene_id.encode("utf-8"))
            protein_embedding = np.load(io.BytesIO(emb_bytes))
            protein_embedding = torch.tensor(protein_embedding, dtype=torch.float32)

        drug_graph = self.smiles_cache[smiles]

        if self.tokenizer and interaction_phrase:
            encoded_text = self.tokenizer(
                interaction_phrase,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            encoded_text = {k: v.squeeze(0) for k, v in encoded_text.items()}
        else:
            encoded_text = None

        return {
            'protein_embedding': protein_embedding,
            'drug_graph': drug_graph,
            'interaction': interaction_phrase,
            'encoded_text': encoded_text,
            'gene_id': gene_id,
            'smiles': smiles
        }

    def extract_interaction_phrase(self, text) -> str:
        """
        Returns the text sandwiched between the two entity tags,
        no matter which order they appear in.  Falls back to an
        empty string so the caller can decide what to do.
        """
        # [DRUG] … [PROTEIN]
        m = re.search(fr"{drug_tag}(.*?){protein_tag}", text)
        if m:
            return m.group(1).strip()

        # [PROTEIN] … [DRUG]
        m = re.search(fr"{protein_tag}(.*?){drug_tag}", text)
        if m:
            return m.group(1).strip()

        # If we get here the line is malformed; return sentinel
        return ""

def collate_fn(batch):
    """Fixed collate function with proper device handling"""
    from torch.nn.utils.rnn import pad_sequence
    protein_embeddings = pad_sequence(
        [item['protein_embedding'] for item in batch],
        batch_first=True
    )
    drug_graphs = [item['drug_graph'] for item in batch]
    interactions = [item['interaction'] for item in batch]
    gene_ids = [item['gene_id'] for item in batch]
    smiles = [item['smiles'] for item in batch]

    batched_drugs = Batch.from_data_list(drug_graphs)
    encoded_texts = None
    if all(item['encoded_text'] is not None for item in batch):
        encoded_texts = {
            k: torch.stack([item['encoded_text'][k] for item in batch])
            for k in batch[0]['encoded_text'].keys()
        }

    return {
        'protein_embeddings': protein_embeddings,
        'drug_graphs': batched_drugs,
        'interactions': interactions,
        'encoded_texts': encoded_texts,
        'gene_ids': gene_ids,
        'smiles': smiles
    }
