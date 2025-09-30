import os, json, time, pickle, random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score

from loader_xattn import ProteinDrugInteractionDataset, collate_fn
from model_xattn import ExpressionDirectionClassifier

# -----------------------------
# Config
# -----------------------------
CSV_PATH     = os.getenv("CSV_PATH", "/mnt/data/direction_lt500.csv")
LMDB_DIR     = os.getenv("LMDB_PATH", "/mnt/data/filtered_lt500.lmdb")
DRUG_CACHE   = os.getenv("DRUG_CACHE", "/mnt/data/graph_cache.pkl")
GNN          = os.getenv("GNN", "gin")
USE_XATTN    = os.getenv("USE_XATTN", "false").lower() == "true"
FREEZE_GNN   = os.getenv("FREEZE_GNN", "true").lower() == "true"
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 16))
EPOCHS       = int(os.getenv("EPOCHS", 100))
LR           = float(os.getenv("LR", 3e-4))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
VAL_FRACTION = float(os.getenv("VAL_FRACTION", 0.2))
SAVE_DIR     = os.getenv("SAVE_DIR", "/mnt/data/cls/runs/direction")
SEED         = int(os.getenv("SEED", 42))
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn_ckpt     = "/mnt/data/gcn_contextpred.pth" if GNN == "gcn" else "/mnt/data/gin_contextpred.pth"
RETURN_SEQS  = os.getenv("RETURN_SEQUENCES", "true").lower() == "true"
MAX_NODES    = int(os.getenv("MAX_NODES", 50))

# Determinism / env
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# -----------------------------
# Data
# -----------------------------
df = pd.read_csv(CSV_PATH)
print(f"[RAW_CSV] Total rows: {len(df)}")
print(f"[RAW_CSV] Columns: {list(df.columns)}")
print(f"[RAW_CSV] target_id unique: {df['target_id'].unique()[:10]}")
print(f"[RAW_CSV] direction_id unique: {df['direction_id'].unique()}")
print(f"[RAW_CSV] direction_id counts before filter: {Counter(df['direction_id'].tolist())}")

# Filter
df = df[(df["target_id"] == 0) & (df["direction_id"].isin([0, 1]))].reset_index(drop=True)
ctr = Counter(df["direction_id"].tolist())
print(f"[FILTERED_CSV] N={len(df)} class_counts={dict(ctr)} pos_ratio={ctr.get(1,0)/max(1,sum(ctr.values())):.4f}")
if len(ctr) == 1: print("CRITICAL: Only one class remains after filtering!")

# Cache
with open(DRUG_CACHE, "rb") as f:
    graph_cache = pickle.load(f)

dataset = ProteinDrugInteractionDataset(
    seg_df=df,
    lmdb_path=LMDB_DIR,
    smiles_cache=graph_cache,
    verbose=False
)

# -----------------------------
# Helper: safe index probe
# -----------------------------
def _safe_item_ok(ds, i):  # <<< ADDED
    try:
        _ = ds[i]
        return True
    except KeyError as e:
        print(f"[SKIP] {e}")
        return False
    except Exception as e:
        print(f"[SKIP-OTHER] idx={i} err={repr(e)}")
        return False

# Quick dataset sanity
print("[VALIDATION] Checking first 10 samples...")
sample_labels = []
for i in range(min(10, len(dataset))):
    try:  # <<< ADDED
        s = dataset[i]  # <<< CHANGED (wrapped in try)
        sample_labels.append(int(s["direction_id"]))
        if torch.isnan(s["protein"]).any(): print(f"WARNING: NaN in protein at sample {i}")
        if torch.isinf(s["protein"]).any(): print(f"WARNING: Inf in protein at sample {i}")
    except Exception as e:  # <<< ADDED
        print(f"[SANITY-SKIP] idx={i} err={repr(e)}")  # <<< ADDED
print(f"[SAMPLE_LABELS] First 10: {sample_labels}")
print(f"[SAMPLE_DISTRIBUTION] {Counter(sample_labels)}")

# Split
idx = np.arange(len(dataset)); np.random.shuffle(idx)
n_val = int(len(idx) * VAL_FRACTION)
val_idx = idx[:n_val] if n_val > 0 else []
train_idx = idx[n_val:] if n_val > 0 else idx
print(f"[SPLIT_CHECK] Train first 100: {Counter(df.iloc[train_idx[:100]]['direction_id'])}")
print(f"[SPLIT_CHECK] Val first 100: {Counter(df.iloc[val_idx[:100]]['direction_id']) if len(val_idx)>0 else {}}")

# Remove problematic samples that raise in __getitem__    # <<< ADDED
train_idx = [int(i) for i in train_idx if _safe_item_ok(dataset, int(i))]  # <<< ADDED
val_idx   = [int(i) for i in val_idx   if _safe_item_ok(dataset, int(i))] if len(val_idx) > 0 else []  # <<< ADDED
print(f"[FILTERED_IDX] train={len(train_idx)} val={len(val_idx)}")  # <<< ADDED

train_set, val_set = Subset(dataset, train_idx), (Subset(dataset, val_idx) if n_val > 0 else None)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=False, collate_fn=collate_fn, drop_last=True
)
val_loader = None if val_set is None else DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=False, collate_fn=collate_fn, drop_last=False
)
print(f"[TRAIN] batch_size={BATCH_SIZE} steps_per_epoch={len(train_loader)} workers=0")

# One batch check
print("[LOADER_CHECK] First training batch distro…")
try:  # <<< ADDED
    _check = next(iter(DataLoader(train_set, batch_size=min(32, len(train_set)),
                                  shuffle=False, num_workers=0, collate_fn=collate_fn)))
    print(f"[BATCH_DISTRIBUTION] {dict(Counter(_check['direction_ids'].tolist()))}")
    del _check
except Exception as e:  # <<< ADDED
    print(f"[LOADER_CHECK-SKIP] err={repr(e)}")  # <<< ADDED

# -----------------------------
# Model / Optim / Loss / Sched
# -----------------------------
model = ExpressionDirectionClassifier(
    dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=gnn_ckpt,
    freeze_gnn=FREEZE_GNN, gnn_type=GNN, use_xattn=USE_XATTN,
    n_classes=2, return_sequences=RETURN_SEQS
).to(DEVICE)

# differential LRs
head_params, backbone_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    (head_params if n.startswith("head.") else backbone_params).append(p)

opt = torch.optim.AdamW(
    [{'params': backbone_params, 'lr': LR * 0.1},
     {'params': head_params,     'lr': LR}],
    weight_decay=WEIGHT_DECAY
)

# Balanced data → plain CE
crit = nn.CrossEntropyLoss()

# We want to optimize AUC upward; ReduceLROnPlateau minimizes → step on negative AUC
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=5, verbose=True
)

def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------
# Metrics (log decision margins)
# -----------------------------
def epoch_metrics(all_logits, all_y):
    if not all_logits:
        return dict(acc=float("nan"), pos_recall=float("nan"), neg_recall=float("nan"),
                    mu=float("nan"), sd=float("nan"), auc=float("nan"), auprc=float("nan"))
    y = torch.cat(all_y).detach().cpu().long()
    logits = torch.cat(all_logits).detach().cpu()           # [N,2]
    margins = (logits[:, 1] - logits[:, 0])                 # decision margin
    probs = logits.softmax(dim=-1)[:, 1]
    preds = logits.argmax(dim=-1)

    acc = (preds == y).float().mean().item()
    c = Counter(y.tolist()); pos = c.get(1, 0); neg = c.get(0, 0)
    pos_recall = ((preds[y == 1] == 1).float().mean().item()) if pos > 0 else float("nan")
    neg_recall = ((preds[y == 0] == 0).float().mean().item()) if neg > 0 else float("nan")
    mu, sd = margins.mean().item(), margins.std().item()
    try:
        y_np, p_np = y.numpy(), probs.numpy()
        auc   = roc_auc_score(y_np, p_np) if len(set(y_np)) > 1 else float("nan")
        auprc = average_precision_score(y_np, p_np) if len(set(y_np)) > 1 else float("nan")
    except Exception:
        auc, auprc = float("nan"), float("nan")
    return dict(acc=acc, pos_recall=pos_recall, neg_recall=neg_recall, mu=mu, sd=sd, auc=auc, auprc=auprc)

# -----------------------------
# Train
# -----------------------------
best_auc = -float("inf"); best_state = None
early_stop_patience = 10
patience_counter = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    tot = 0.0; nb = 0
    all_logits_t, all_y_t = [], []

    for batch_idx, b in enumerate(train_loader):
        p  = b["protein_embeddings"].to(DEVICE, non_blocking=True)
        pm = b["protein_pad_mask"].to(DEVICE, non_blocking=True)
        y  = b["direction_ids"].to(DEVICE, non_blocking=True).long()

        if ep <= 3 and batch_idx == 0:
            print(f"[E{ep}_B{batch_idx}] batch_labels={dict(Counter(y.cpu().tolist()))}")

        opt.zero_grad(set_to_none=True)
        logits = model(p_vec=p, batch_data=b["drug_graphs"], p_mask=pm)
        loss = crit(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss at batch {batch_idx}: {loss.item()}"); cleanup_memory(); continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tot += loss.item(); nb += 1
        all_logits_t.append(logits.detach()); all_y_t.append(y.detach())

        if batch_idx % 50 == 0:
            cleanup_memory()

    train_loss = tot / max(1, nb)
    train_m = epoch_metrics(all_logits_t, all_y_t)

    # Eval
    if val_loader is not None:
        model.eval()
        vtot = 0.0; vnb = 0
        all_logits_v, all_y_v = [], []
        with torch.no_grad():
            for b in val_loader:
                p  = b["protein_embeddings"].to(DEVICE, non_blocking=True)
                pm = b["protein_pad_mask"].to(DEVICE, non_blocking=True)
                y  = b["direction_ids"].to(DEVICE, non_blocking=True).long()
                logits = model(p_vec=p, batch_data=b["drug_graphs"], p_mask=pm)
                loss = crit(logits, y)
                vtot += loss.item(); vnb += 1
                all_logits_v.append(logits); all_y_v.append(y)
        val_loss = vtot / max(1, vnb)
        val_m = epoch_metrics(all_logits_v, all_y_v)
    else:
        val_loss, val_m = train_loss, train_m

    # Scheduler & early-stop on AUC (maximize AUC → minimize -AUC)
    cur_auc = val_m["auc"] if val_m["auc"] == val_m["auc"] else -float("inf")  # handle NaN
    scheduler.step(-cur_auc)

    dt = time.time() - t0
    print(
        f"[E{ep}] "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_acc={train_m['acc']:.3f} val_acc={val_m['acc']:.3f} "
        f"posR={train_m['pos_recall']:.3f}/{val_m['pos_recall']:.3f} "
        f"negR={train_m['neg_recall']:.3f}/{val_m['neg_recall']:.3f} "
        f"margin_mu,sd={val_m['mu']:.3f},{val_m['sd']:.3f} "
        f"AUC/AUPRC={val_m['auc']:.3f}/{val_m['auprc']:.3f} "
        f"time_s={dt:.1f}"
    )

    if ep in (1, 2, 5) or ep % 10 == 0:
        with torch.no_grad():
            g = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    g.append((name, p.grad.norm().item()))
            g = sorted(g, key=lambda x: -x[1])[:8]
            print("[GRADS top8]", g)

    if cur_auc > best_auc:
        best_auc = cur_auc
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if ep > 15 and cur_auc < 0.55:
        print(f"[EARLY STOP] AUC not improving sufficiently after {ep} epochs")
        break
    if patience_counter >= early_stop_patience:
        print(f"[EARLY STOP] No AUC improvement for {early_stop_patience} epochs")
        break

# Save best
if best_state is not None:
    model.load_state_dict(best_state)

ckpt_path = os.path.join(SAVE_DIR, "model.pt")
torch.save(model.state_dict(), ckpt_path)
with open(os.path.join(SAVE_DIR, "run.json"), "w") as f:
    json.dump({"seed": SEED, "device": str(DEVICE), "best_auc": float(best_auc)}, f, indent=2)
print(f"[SAVED] {ckpt_path} best_auc={best_auc:.4f}")
