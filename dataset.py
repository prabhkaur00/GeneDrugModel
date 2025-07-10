from torch.utils.data import Dataset
import pandas as pd
import h5py
import pickle
import os
import torch
from torch_geometric.data import Batch
from utils import cache_to_pyg_data

class ProteinDrugDataset(Dataset):
    def __init__(self, h5_file, csv_subset, smiles_cache, tokenizer=None, max_len=25):
        self.data_df = csv_subset
        self.h5_file_path = h5_file
        self.smiles_cache = smiles_cache
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.valid_indices = []
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            keys = set(h5_file.keys())
            for idx, row in self.data_df.iterrows():
                gene_id = str(int(row['GeneID']))
                smiles = row['SMILES']
                h5_key = f"genes_{gene_id}"
                if h5_key in keys and smiles in self.smiles_cache:
                    self.valid_indices.append(idx)

        print(f"Stage dataset loaded: {len(self.valid_indices)} valid entries from {os.path.basename(self.h5_file_path)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        df_idx = self.valid_indices[idx]
        row = self.data_df.iloc[df_idx]
        gene_id = str(int(row['GeneID']))
        smiles = row['SMILES']
        interaction = row['Interaction']

        interaction_phrase = self.extract_interaction_phrase(interaction)

        # CHANGED: always load from a single HDF5 file path
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            protein_embedding = torch.tensor(h5_file[f"genes_{gene_id}"][:]).float()

        drug_graph = cache_to_pyg_data(self.smiles_cache[smiles])

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

    def extract_interaction_phrase(self, text):
        try:
            before, after_drug = text.split("[DRUG]", 1)
            phrase, _ = after_drug.split("[PROTEIN]", 1)
            return phrase.strip()
        except Exception as e:
            msg = f"[extract_interaction_phrase] Fallback used for: {text} | Error: {str(e)}"
            print(msg)
            with open("logs/error_log.txt", "a") as f: f.write(msg + "\n")
            return "affects the expression"

def collate_fn(batch):
    """Fixed collate function with proper device handling"""
    max_protein_len = max([item['protein_embedding'].shape[0] for item in batch])

    padded_proteins = []
    for item in batch:
        protein_emb = item['protein_embedding']
        current_len = protein_emb.shape[0]

        if current_len < max_protein_len:
            padding = torch.zeros(max_protein_len - current_len, protein_emb.shape[1], dtype=protein_emb.dtype)
            padded_protein = torch.cat([protein_emb, padding], dim=0)
        else:
            padded_protein = protein_emb[:max_protein_len]
        padded_proteins.append(padded_protein)

    protein_embeddings = torch.stack(padded_proteins)
    drug_graphs = [item['drug_graph'] for item in batch]
    interactions = [item['interaction'] for item in batch]
    gene_ids = [item['gene_id'] for item in batch]
    smiles = [item['smiles'] for item in batch]

    # Batch drug graphs
    batched_drugs = Batch.from_data_list(drug_graphs)

    # Handle encoded texts
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
