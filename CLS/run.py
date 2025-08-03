import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from collections import Counter
import torch.nn.functional as F
import multiprocessing
from pathlib import Path

# -------------------- CONFIG --------------------
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 128))
EPOCHS       = int(os.getenv("EPOCHS", 10))
LR           = float(os.getenv("LR", 1e-4))
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", 0))
PREFETCH_FACTOR = os.getenv("PREFETCH_FACTOR")
PREFETCH_FACTOR = int(PREFETCH_FACTOR) if PREFETCH_FACTOR is not None else None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"config: BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, NUM_WORKERS={NUM_WORKERS}, DEVICE={DEVICE}")
SEG_DF_PATH = '/mnt/data/cls/segments_2head.csv'
VOCAB_PATH  = '/mnt/data/cls/vocab_2head.json'
LMDB_PATH   = '/mnt/data/gene_data/mean-pooled-all.lmdb'
DRUG_CACHE  = '/mnt/data/graph_cache.pkl'
gnn_ckpt = '/mnt/data/gcn_contextpred.pth'
ckpt_dir = 'mnt/data/cls/checkpoints'

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

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss(loss_type, weight=None, gamma=2.0):
    if loss_type == "cross_entropy":
        return torch.nn.CrossEntropyLoss(weight=weight)
    elif loss_type == "focal":
        return FocalLoss(weight=weight, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# -------------------- LOAD DATA --------------------
with open(DRUG_CACHE, "rb") as f:
    graph_cache = pickle.load(f)

LOG_FILE = "/mnt/data/cls/logs.txt"

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
N = len(dataset)
train_size = int(0.8 * N)
val_size = N - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

all_target_ids = [train_set[i]["target_id"] for i in range(len(train_set))]
target_counts = Counter(all_target_ids)
target_weights = {cls: 1.0 / (count + 1e-6) for cls, count in target_counts.items()}
sample_weights = [target_weights[train_set[i]["target_id"]] for i in range(len(train_set))]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_set),
    replacement=True
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR
)

# -------------------- MODEL --------------------
# Count class frequencies in train set
all_targets = []
all_directions = []
for i in range(len(train_set)):
    item = train_set[i]
    all_targets.append(item["target_id"])
    all_directions.append(item["direction_id"])

target_counts = Counter(all_targets)
direction_counts = Counter(all_directions)

def compute_class_weights(counter, num_classes):
    freq = torch.tensor([counter.get(i, 0) for i in range(num_classes)], dtype=torch.float)
    freq = freq.clamp(min=1.0)
    weights = torch.log(freq.sum() / freq)
    weights = weights / weights.sum() * num_classes
    return weights.to(DEVICE)

model = TwoHeadConditional(
    dim_p=768, emb_dim=300,
    d_model=512,
    n_targets=NUM_T,
    n_dirs=NUM_D,
    gnn_ckpt = gnn_ckpt
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
target_weights = compute_class_weights(target_counts, NUM_T)
direction_weights = compute_class_weights(direction_counts, NUM_D)
crit_t = torch.nn.CrossEntropyLoss(weight=target_weights)
crit_d = torch.nn.CrossEntropyLoss(weight=direction_weights)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_config = {
    "type": "focal",
    "gamma": 2.0
} ### loss_config = {"type": "cross_entropy"}
target_weights = compute_class_weights(target_counts, NUM_T)
direction_weights = compute_class_weights(direction_counts, NUM_D)
gamma = loss_config["gamma"] if loss_config["type"] == "focal" else None
crit_t = get_loss(
    loss_type=loss_config["type"],
    weight=target_weights.to(DEVICE),
    gamma=gamma
)
crit_d = get_loss(
    loss_type=loss_config["type"],
    weight=direction_weights.to(DEVICE),
    gamma=gamma
)

def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[PARAMS] Total trainable: {total:,}")
    return total

count_trainable_params(model)

best_f1 = 0
best_epoch = -1
# -------------------- TRAINING LOOP --------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        p_vec = batch["protein_embeddings"]  # (B, 768)
        d_vec = batch["drug_graphs"]         # PyG Batch object
        y_t   = batch["target_ids"]          # (B,)
        y_d   = batch["direction_ids"]       # (B,)
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
    report_t, report_d = evaluate(val_loader)
    print("Target Head Report:\n", report_t)
    print("Direction Head Report:\n", report_d)
# ---------------------------------------------------------