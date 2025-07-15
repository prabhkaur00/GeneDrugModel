import os
import pickle
import torch
import pandas as pd
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import ProteinDrugDataset
from model import ProteinDrugLLMModel
from train import train_model
import sys
import random
import numpy as np
import h5py
import argparse
import time

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--flag", type=str, choices=["local"], help="Use Google Drive paths if flag is 'local'")
args = parser.parse_args()

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)         # safer with h5py
mp.set_sharing_strategy("file_system")           # avoid /dev/shm limit
colab = args.flag == "local" 

if colab:
    h5_dir = '/content/drive/MyDrive/Shared-with-booknerd'
    csv_file = '/content/drive/MyDrive/Shared-with-booknerd/tier1_smiles_smol.csv'
    smiles_cache_file = '/content/drive/MyDrive/Shared-with-booknerd/smiles_cache.pkl'
    model_name = 'meta-llama/Llama-2-7b-hf'
    gnn_ckpt = '/content/drive/MyDrive/Shared-with-booknerd/gcn_contextpred.pth'
    graph_cache_file = '/content/drive/MyDrive/Shared-with-booknerd/graph_cache.pkl'
else:
    h5_dir = '/mnt/data/gene_data'
    csv_file = '/mnt/data/tier1.csv'
    smiles_cache_file = '/mnt/data/smiles_cache.pkl'
    graph_cache_file = '/mnt/data/graph_cache.pkl'
    model_name = '/mnt/data/vicuna-13b-v1.5'
    gnn_ckpt = '/mnt/data/gcn_contextpred.pth'

print("Loading tokenizer")
t0 = time.time(); print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
print(f"Tokenizer: {time.time()-t0:.1f}s")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["[PROTEIN]", "[DRUG]"]})
print("Added special tokens")

print("Loading full CSV and SMILES cache")
full_csv_df = pd.read_csv(csv_file)
import pickle, tqdm
from utils import cache_to_pyg_data

with open(smiles_cache_file, "rb") as f:
    smiles_cache_raw = pickle.load(f)
graph_cache = {}
for k, raw in tqdm.tqdm(smiles_cache_raw.items(), desc="Building PyG graphs"):
    graph_cache[k] = cache_to_pyg_data(raw)   # → torch_geometric.data.Data
with open(graph_cache_file, "wb") as f:
    pickle.dump(graph_cache, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProteinDrugLLMModel(
    llm_model_name=model_name,
    tokenizer=tokenizer,
    gnn_ckpt=gnn_ckpt,
    freeze_gnn=True
)

print("Model created")

h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
for epoch in range(1, 2):
    print(f"\n=== Epoch {epoch} ===")
    for h5_fname in h5_files:
        print(f"\n--- Training on stage: {h5_fname} ---")
        stage_start = time.time()
        h5_path = os.path.join(h5_dir, h5_fname)
        print("Loading H5 keys...")
        with h5py.File(h5_path, 'r') as h5_file:
            keys = set(h5_file.keys())

        print("Filtering CSV rows...")
        relevant_rows = []
        for idx, row in full_csv_df.iterrows():
            gene_id = str(int(row['GeneID']))
            h5_key = f"genes_{gene_id}"
            if h5_key in keys and row['SMILES'] in graph_cache:
                relevant_rows.append(row)
        print(f"Filtered to {len(relevant_rows)} rows.")

        print("Building dataset...")
        csv_subset = pd.DataFrame(relevant_rows).reset_index(drop=True)
        dataset = ProteinDrugDataset(
            h5_file=h5_path,
            csv_subset=csv_subset,
            graph_cache=graph_cache,
            tokenizer=tokenizer
        )

        print("Splitting data...")
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=epoch)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        print("Training model...")
        trained_model = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            batch_size=1,
            log_predictions=True,
            log_frequency=5,
            num_epochs=1
        )

        ckpt_path = f"/mnt/data/checkpoints/model_epoch{epoch}_{h5_fname}.pt"
        print(f"Saving checkpoint to {ckpt_path}...")
        torch.save(model.state_dict(), ckpt_path)

        print(f"Stage {h5_fname} done in {time.time() - stage_start:.2f}s")