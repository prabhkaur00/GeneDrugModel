#!/usr/bin/env python3
import os, json, pickle, datetime
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from collections import Counter
import pandas as pd
from model import StageBJoint
from loader import ProteinDrugInteractionDataset, collate_fn

# -------------------- CONFIG (env-driven) --------------------
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 128))
EPOCHS       = int(os.getenv("EPOCHS", 10))
LR           = float(os.getenv("LR", 3e-4))               # good starting LR
LR_MIN       = float(os.getenv("LR_MIN", 3e-5))           # floor for cosine
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
WARMUP_FRAC  = float(os.getenv("WARMUP_FRAC", 0.05))      # 5% warmup
IMB_STRAT    = os.getenv("IMBALANCE_STRATEGY", "focal")   # focal|weights|sampler|none
FOCAL_GAMMA  = float(os.getenv("FOCAL_GAMMA", 1.75))      # easy to sweep
LAMBDA_T     = float(os.getenv("LAMBDA_T", 1.0))          # target head weight
LAMBDA_D     = float(os.getenv("LAMBDA_D", 1.0))          # direction head weight

USE_XATTN    = os.getenv("USE_XATTN", "true").lower()=="true"
FREEZE_GNN   = os.getenv("FREEZE_GNN", "true").lower()=="true"
GNN          = os.getenv("GNN", "gin")
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", 0))
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR")) if os.getenv("PREFETCH_FACTOR") else None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_DF_PATH = os.getenv("SEG_DF_PATH", "/mnt/data/cls/segments_2head.csv")
VOCAB_PATH  = os.getenv("VOCAB_PATH",  "/mnt/data/cls/vocab_minimal.json")
LMDB_PATH   = os.getenv("LMDB_PATH",   "/mnt/data/gene_data/lmdb_parts")
DRUG_CACHE  = os.getenv("DRUG_CACHE",  "/mnt/data/graph_cache.pkl")
gnn_ckpt    = '/mnt/data/gcn_contextpred.pth' if GNN=="gcn" else '/mnt/data/gin_contextpred.pth'
LOG_FILE    = os.getenv("LOG_FILE", "/mnt/data/cls/logs.txt")

print(f"cfg: BS={BATCH_SIZE} E={EPOCHS} LR={LR} WD={WEIGHT_DECAY} warmup={WARMUP_FRAC} "
      f"gamma={FOCAL_GAMMA} imb={IMB_STRAT} xattn={USE_XATTN} freeze_gnn={FREEZE_GNN} "
      f"nw={NUM_WORKERS} prefetch={PREFETCH_FACTOR} device={DEVICE}")

# -------------------- UTILS --------------------
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full = f"[{ts}] {msg}"
    print(full, flush=True)
    with open(LOG_FILE, "a") as f: f.write(full + "\n")

def compute_class_weights(counter, num_classes, device):
    freq = torch.tensor([counter.get(i, 0) for i in range(num_classes)], dtype=torch.float)
    freq = freq.clamp(min=1.0)
    w = torch.log(freq.sum() / freq)
    return (w / w.sum() * num_classes).to(device)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.gamma = gamma
    def forward(self, input, target):
        ce = self.ce(input, target)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

def make_loss(kind, weight=None, gamma=2.0):
    if kind == "cross_entropy": return nn.CrossEntropyLoss(weight=weight)
    if kind == "focal":        return FocalLoss(weight=weight, gamma=gamma)
    raise ValueError(f"Unsupported loss: {kind}")

def evaluate(model, loader, device, target_names, dir_names):
    model.eval()
    preds_t, preds_d, true_t, true_d = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            p, d = batch["protein_embeddings"].to(device), batch["drug_graphs"].to(device)
            y_t, y_d = batch["target_ids"].to(device), batch["direction_ids"].to(device)
            logit_t, logit_d = model(p, d)
            preds_t.extend(torch.argmax(logit_t, -1).cpu().tolist())
            preds_d.extend(torch.argmax(logit_d, -1).cpu().tolist())
            true_t.extend(y_t.cpu().tolist())
            true_d.extend(y_d.cpu().tolist())
    rt = classification_report(true_t, preds_t, target_names=target_names, zero_division=0)
    rd = classification_report(true_d, preds_d, target_names=dir_names, zero_division=0)
    return rt, rd

def build_scheduler(optimizer, total_steps, warmup_frac, lr_min, base_lr):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # cosine to lr_min/base_lr
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * t))).item()
        # scale so end hits lr_min/base_lr
        end_ratio = lr_min / base_lr
        return end_ratio + (1 - end_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# -------------------- DATA --------------------
with open(DRUG_CACHE, "rb") as f: graph_cache = pickle.load(f)
seg_df = pd.read_csv(SEG_DF_PATH)

with open(VOCAB_PATH) as f:
    vocabs = json.load(f)
target2id = vocabs["target2id"]; direction2id = vocabs["direction2id"]
NUM_T, NUM_D = len(target2id), len(direction2id)

dataset = ProteinDrugInteractionDataset(seg_df, LMDB_PATH, graph_cache)
train_set = ProteinDrugInteractionDataset(
    tr_df, LMDB_PATH, graph_cache,
    gid2lmdb_json="/mnt/data/geneid_to_lmdb.json"
)
val_set = ProteinDrugInteractionDataset(
    va_df, LMDB_PATH, graph_cache,
    gid2lmdb_json="/mnt/data/geneid_to_lmdb.json"
)
test_set = ProteinDrugInteractionDataset(
    te_df, LMDB_PATH, graph_cache,
    gid2lmdb_json="/mnt/data/geneid_to_lmdb.json"
)
targets = [train_set[i]["target_id"] for i in range(len(train_set))]
dirs    = [train_set[i]["direction_id"] for i in range(len(train_set))]
t_counts, d_counts = Counter(targets), Counter(dirs)
sample_weights = [1.0 / (t_counts[t] + 1e-6) for t in targets]

if IMB_STRAT == "sampler":
    sampler, cw_t, cw_d, loss_kind = WeightedRandomSampler(sample_weights, len(train_set), True), None, None, "cross_entropy"
elif IMB_STRAT == "weights":
    sampler, cw_t, cw_d, loss_kind = None, compute_class_weights(t_counts, NUM_T, DEVICE), compute_class_weights(d_counts, NUM_D, DEVICE), "cross_entropy"
elif IMB_STRAT == "focal":
    sampler, cw_t, cw_d, loss_kind = None, None, None, "focal"
else:
    sampler, cw_t, cw_d, loss_kind = None, None, None, "cross_entropy"

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None),
    collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0), prefetch_factor=PREFETCH_FACTOR
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0), prefetch_factor=PREFETCH_FACTOR
)

# -------------------- MODEL --------------------
model = StageBJoint(
    dim_p=768, emb_dim=300, d_model=512, n_dirs=NUM_D,
    gnn_ckpt=gnn_ckpt, freeze_gnn=FREEZE_GNN, use_xattn=USE_XATTN
).to(DEVICE)

# -------------------- OPTIM + SCHED --------------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = max(1, EPOCHS * len(train_loader))
sched = build_scheduler(opt, total_steps, WARMUP_FRAC, LR_MIN, LR)

# -------------------- LOSSES --------------------
crit_t = make_loss(loss_kind, weight=cw_t, gamma=FOCAL_GAMMA)
crit_d = make_loss(loss_kind, weight=cw_d, gamma=FOCAL_GAMMA)

log(f"[PARAMS] Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
log(f"[LOSS] kind={loss_kind} gamma={FOCAL_GAMMA} λ_t={LAMBDA_T} λ_d={LAMBDA_D}")

# -------------------- TRAIN --------------------
global_step = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        p, d = batch["protein_embeddings"].to(DEVICE), batch["drug_graphs"].to(DEVICE)
        y_t, y_d = batch["target_ids"].to(DEVICE), batch["direction_ids"].to(DEVICE)

        opt.zero_grad(set_to_none=True)
        logit_t, logit_d = model(p, d)
        loss = LAMBDA_T * crit_t(logit_t, y_t) + LAMBDA_D * crit_d(logit_d, y_d)
        loss.backward()
        opt.step()
        sched.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            for i, pg in enumerate(opt.param_groups):
                if "lr" in pg: cur_lr = pg["lr"]; break
            log(f"[E{epoch+1} B{batch_idx}] lr={cur_lr:.3e} "
                f"Loss_t={crit_t(logit_t, y_t).item():.4f} "
                f"Loss_d={crit_d(logit_d, y_d).item():.4f} Total={loss.item():.4f}")
        global_step += 1

    log(f"[EPOCH {epoch+1}/{EPOCHS}] AvgLoss={total_loss/len(train_loader):.4f}")
    rt, rd = evaluate(model, val_loader, DEVICE, list(target2id.keys()), list(direction2id.keys()))
    print("Target Report:\n", rt)
    print("Direction Report:\n", rd)