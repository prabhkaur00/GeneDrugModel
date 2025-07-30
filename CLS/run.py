import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model import TwoHeadConditional               # your model
from loader import ProteinDrugInteractionDataset  # your dataset
from loader import collate_fn                     # your collate_fn
import pandas as pd
import json
import os
import pickle
import tqdm
from torch_geometric.data import Data, Batch
import datetime
# -------------------- CONFIG --------------------
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEG_DF_PATH = "/content/drive/MyDrive/GeneDrugChat-Data/analysis_outputs/segments_2head.csv"
VOCAB_PATH  = "/content/drive/MyDrive/GeneDrugChat-Data/analysis_outputs/vocab_2head.json"
LMDB_PATH   = '/content/local_lmdb'
DRUG_CACHE  = '/content/drive/MyDrive/GeneDrugChat-Data/smiles_cache.pkl'
gnn_ckpt = '/content/drive/MyDrive/GeneDrugChat-Data/gcn_contextpred.pth'

def cache_to_pyg_data(graph_dict):
    """Convert cached graph to PyG Data object"""
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
    edge_feat = torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
    node_feat = torch.tensor(graph_dict['node_feat'], dtype=torch.long)
    num_nodes = graph_dict['num_nodes']

    data = Data(
        x=node_feat,
        edge_index=edge_index,
        edge_attr=edge_feat,
        num_nodes=num_nodes
    )
    return data

with open(DRUG_CACHE, "rb") as f:
    smiles_cache_raw = pickle.load(f)
graph_cache = {}
for k, raw in tqdm.tqdm(smiles_cache_raw.items(), desc="Building PyG graphs"):
    graph_cache[k] = cache_to_pyg_data(raw) 

LOG_FILE = "/content/logs.txt"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# -------------------- LOAD DATA --------------------
seg_df = pd.read_csv(SEG_DF_PATH)

with open(VOCAB_PATH) as f:
    vocabs = json.load(f)
target2id = vocabs["target2id"]
direction2id = vocabs["direction2id"]
NUM_T = len(target2id)
NUM_D = len(direction2id)


dataset = ProteinDrugInteractionDataset(
    seg_df=seg_df,
    protein_lmdb_path=LMDB_PATH,
    smiles_cache=graph_cache
)

# Split
N = len(dataset)
train_size = int(0.8 * N)
val_size = N - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
# ----------------------------------------------------

# -------------------- MODEL --------------------
model = TwoHeadConditional(
    dim_p=768, emb_dim=300,
    d_model=512,
    n_targets=NUM_T,
    n_dirs=NUM_D,
    gnn_ckpt = gnn_ckpt
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
crit_t = torch.nn.CrossEntropyLoss()
crit_d = torch.nn.CrossEntropyLoss()

def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[PARAMS] Total trainable: {total:,}")
    return total
# ------------------------------------------------

# -------------------- TRAINING LOOP --------------------
def evaluate(loader):
    model.eval()
    all_preds_t, all_preds_d = [], []
    all_true_t, all_true_d = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            p = batch["protein_embeddings"]  # (B, 768)
            d = batch["drug_graphs"]         # PyG Batch object
            y_t   = batch["target_ids"]          # (B,)
            y_d   = batch["direction_ids"]       # (B,)

            p, d = p.to(DEVICE), d.to(DEVICE)
            y_t, y_d = y_t.to(DEVICE), y_d.to(DEVICE)
            logit_t, logit_d = model(p, d)
            pred_t = torch.argmax(logit_t, dim=-1)
            pred_d = torch.argmax(logit_d, dim=-1)

            all_preds_t.extend(pred_t.cpu().tolist())
            all_preds_d.extend(pred_d.cpu().tolist())
            all_true_t.extend(y_t.cpu().tolist())
            all_true_d.extend(y_d.cpu().tolist())

    report_t = classification_report(all_true_t, all_preds_t, target_names=target2id.keys(), zero_division=0)
    report_d = classification_report(all_true_d, all_preds_d, target_names=direction2id.keys(), zero_division=0)

    return report_t, report_d

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        p_vec = batch["protein_embeddings"]  # (B, 768)
        d_vec = batch["drug_graphs"]         # PyG Batch object
        y_t   = batch["target_ids"]          # (B,)
        y_d   = batch["direction_ids"]       # (B,)

        # log(f"[BATCH {batch_idx}] p_vec shape: {p_vec.shape}")
        # log(f"[BATCH {batch_idx}] d_vec.num_graphs: {d_vec.num_graphs}")
        # log(f"[BATCH {batch_idx}] y_t shape: {y_t.shape}, y_d shape: {y_d.shape}")

        p_vec, d_vec = p_vec.to(DEVICE), d_vec.to(DEVICE)
        y_t, y_d = y_t.to(DEVICE), y_d.to(DEVICE)

        opt.zero_grad()
        logit_t, logit_d = model(p_vec, d_vec)

        loss_t = crit_t(logit_t, y_t)
        loss_d = crit_d(logit_d, y_d)
        loss = loss_t + loss_d

        if batch_idx % 100 == 0:
            log(f"[E{epoch+1} B{batch_idx}] Loss_t: {loss_t.item():.4f}, "
                f"Loss_d: {loss_d.item():.4f}, Total: {loss.item():.4f}")

        loss.backward()
        opt.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"\n[EPOCH {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # Eval
    report_t, report_d = evaluate(val_loader)
    print("Target Head Report:\n", report_t)
    print("Direction Head Report:\n", report_d)
# ---------------------------------------------------------