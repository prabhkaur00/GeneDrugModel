import os, json, time, pickle, random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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
ATTN_DROPOUT = float(os.getenv("ATTN_DROPOUT", "0.10"))
LOGIT_L2     = float(os.getenv("LOGIT_L2", "1e-4"))
WARMUP_FRAC  = float(os.getenv("WARMUP_FRAC", "0.05"))
EARLY_PATIENCE = int(os.getenv("EARLY_PATIENCE", "5"))

SPLIT_MODE   = os.getenv("SPLIT_MODE", "random").lower()         # random | cold_gene | cold_drug
SPLIT_SEED   = int(os.getenv("SPLIT_SEED", str(SEED)))

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

# -----------------------------
# Helpers for entity columns & cold splits
# -----------------------------
def _detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

GENE_COL = _detect_col(df, ["gene_id","GeneID","gene","target_gene"])
DRUG_COL = _detect_col(df, ["smiles","SMILES","drug","drug_id","DrugID","cid"])
print(f"[COLUMN_DETECT] gene_col={GENE_COL} drug_col={DRUG_COL}")

if SPLIT_MODE in ("cold_gene", "cold_drug") and (GENE_COL is None or DRUG_COL is None):
    raise ValueError(f"SPLIT_MODE={SPLIT_MODE} requires gene & drug columns. "
                     f"Detected gene={GENE_COL}, drug={DRUG_COL}")

def _cold_entity_split(df, entity_col, val_fraction, rng):
    entities = df[entity_col].dropna().astype(str).unique().tolist()
    rng.shuffle(entities)
    target_rows = int(len(df) * val_fraction)
    picked, running = [], 0
    df_counts = df.groupby(entity_col).size().to_dict()
    for e in entities:
        c = df_counts.get(e, 0)
        if c == 0:
            continue
        picked.append(e)
        running += c
        if running >= target_rows:
            break
    picked_set = set(map(str, picked))
    val_mask = df[entity_col].astype(str).isin(picked_set)
    val_idx = df.index[val_mask].to_numpy()
    train_idx = df.index[~val_mask].to_numpy()
    return train_idx, val_idx, picked

# Cache
with open(DRUG_CACHE, "rb") as f:
    graph_cache = pickle.load(f)

dataset = ProteinDrugInteractionDataset(
    seg_df=df,
    lmdb_path=LMDB_DIR,
    smiles_cache=graph_cache,
    verbose=False
)

# Map DF row index -> dataset position
dfidx_to_pos = {dfi: pos for pos, dfi in enumerate(dataset.valid_indices)}
def pos_to_dfidx(pos_idx):
    return [dataset.valid_indices[i] for i in pos_idx]

# Quick dataset sanity
print("[VALIDATION] Checking first 10 samples...")
sample_labels = []
for i in range(min(10, len(dataset))):
    s = dataset[i]
    sample_labels.append(int(s["direction_id"]))
    if torch.isnan(s["protein"]).any(): print(f"WARNING: NaN in protein at sample {i}")
    if torch.isinf(s["protein"]).any(): print(f"WARNING: Inf in protein at sample {i}")
print(f"[SAMPLE_LABELS] First 10: {sample_labels}")
print(f"[SAMPLE_DISTRIBUTION] {Counter(sample_labels)}")

# -----------------------------
# Split
# -----------------------------
rng = random.Random(SPLIT_SEED)
if SPLIT_MODE == "random":
    idx_pos = np.arange(len(dataset))
    rng.shuffle(idx_pos.tolist())
    n_val = int(len(idx_pos) * VAL_FRACTION)
    val_pos = idx_pos[:n_val] if n_val > 0 else np.array([], dtype=int)
    train_pos = idx_pos[n_val:] if n_val > 0 else idx_pos
    train_dfidx = pos_to_dfidx(train_pos)
    val_dfidx   = pos_to_dfidx(val_pos)
    split_meta = {
        "mode": "random",
        "val_fraction": VAL_FRACTION,
        "num_train": int(len(train_pos)),
        "num_val": int(len(val_pos)),
    }
elif SPLIT_MODE == "cold_gene":
    train_dfidx, val_dfidx, heldout_genes = _cold_entity_split(df, GENE_COL, VAL_FRACTION, rng)
    train_pos = np.array([dfidx_to_pos[i] for i in train_dfidx if i in dfidx_to_pos], dtype=int)
    val_pos   = np.array([dfidx_to_pos[i] for i in val_dfidx   if i in dfidx_to_pos], dtype=int)
    tr_genes = set(df.loc[train_dfidx, GENE_COL].astype(str))
    va_genes = set(df.loc[val_dfidx, GENE_COL].astype(str))
    leak_g = len(tr_genes & va_genes)
    split_meta = {
        "mode": "cold_gene",
        "val_fraction": VAL_FRACTION,
        "heldout_genes_count": len(set(heldout_genes)),
        "heldout_genes_sample": list(set(heldout_genes))[:20],
        "gene_overlap_count": leak_g,
        "num_train": int(len(train_pos)),
        "num_val": int(len(val_pos)),
    }
    print(f"[SPLIT] cold_gene: heldout_genes={len(set(heldout_genes))} overlap={leak_g}")
elif SPLIT_MODE == "cold_drug":
    train_dfidx, val_dfidx, heldout_drugs = _cold_entity_split(df, DRUG_COL, VAL_FRACTION, rng)
    train_pos = np.array([dfidx_to_pos[i] for i in train_dfidx if i in dfidx_to_pos], dtype=int)
    val_pos   = np.array([dfidx_to_pos[i] for i in val_dfidx   if i in dfidx_to_pos], dtype=int)
    tr_drugs = set(df.loc[train_dfidx, DRUG_COL].astype(str))
    va_drugs = set(df.loc[val_dfidx, DRUG_COL].astype(str))
    leak_d = len(tr_drugs & va_drugs)
    split_meta = {
        "mode": "cold_drug",
        "val_fraction": VAL_FRACTION,
        "heldout_drugs_count": len(set(heldout_drugs)),
        "heldout_drugs_sample": list(set(heldout_drugs))[:20],
        "drug_overlap_count": leak_d,
        "num_train": int(len(train_pos)),
        "num_val": int(len(val_pos)),
    }
    print(f"[SPLIT] cold_drug: heldout_drugs={len(set(heldout_drugs))} overlap={leak_d}")
else:
    raise ValueError(f"Unsupported SPLIT_MODE={SPLIT_MODE}")

print(f"[SPLIT_CHECK] Train size={len(train_pos)} Val size={len(val_pos)}")
print(f"[SPLIT_CHECK] Train first 100: {Counter(df.iloc[train_dfidx[:100]]['direction_id'])}")
print(f"[SPLIT_CHECK] Val first 100: {Counter(df.iloc[val_dfidx[:100]]['direction_id']) if len(val_pos)>0 else {}}")

train_set = Subset(dataset, train_pos)
val_set   = Subset(dataset, val_pos) if len(val_pos) > 0 else None

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
_check = next(iter(DataLoader(train_set, batch_size=min(32, len(train_set)), shuffle=False, num_workers=0, collate_fn=collate_fn)))
print(f"[BATCH_DISTRIBUTION] {dict(Counter(_check['direction_ids'].tolist()))}")
del _check

# -----------------------------
# Model / Optim / Loss / Sched
# -----------------------------
model = ExpressionDirectionClassifier(
    dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=gnn_ckpt,
    freeze_gnn=FREEZE_GNN, gnn_type=GNN, use_xattn=USE_XATTN,
    n_classes=2, return_sequences=RETURN_SEQS,
    attn_dropout=ATTN_DROPOUT
).to(DEVICE)

head_params, projattn_params, backbone_params = [], [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if n.startswith("head."):
        head_params.append(p)
    elif any(k in n for k in [
        "pproj", "dproj", "xblk.sa.in_proj", "xblk.ca.in_proj",
        "xblk.sa.out_proj", "xblk.ca.out_proj"
    ]):
        projattn_params.append(p)
    else:
        backbone_params.append(p)

opt = torch.optim.AdamW(
    [
      {'params': backbone_params,  'lr': LR * 0.1},
      {'params': projattn_params,  'lr': LR / 3.0},
      {'params': head_params,      'lr': LR},
    ],
    weight_decay=WEIGHT_DECAY
)

crit = nn.CrossEntropyLoss()

warmup_epochs = max(1, int(EPOCHS * WARMUP_FRAC))
main_epochs   = max(1, EPOCHS - warmup_epochs)
scheduler = SequentialLR(
    opt,
    schedulers=[
        LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(opt, T_max=main_epochs)
    ],
    milestones=[warmup_epochs]
)

def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------
# Metrics
# -----------------------------
def epoch_metrics(all_logits, all_y):
    if not all_logits:
        return dict(acc=float("nan"), pos_recall=float("nan"), neg_recall=float("nan"),
                    mu=float("nan"), sd=float("nan"), auc=float("nan"), auprc=float("nan"))
    y = torch.cat(all_y).detach().cpu().long()
    logits = torch.cat(all_logits).detach().cpu()
    margins = (logits[:, 1] - logits[:, 0])
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
early_stop_patience = EARLY_PATIENCE
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
        loss = loss + LOGIT_L2 * (logits ** 2).mean()

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

    cur_auc = val_m["auc"] if val_m["auc"] == val_m["auc"] else -float("inf")
    scheduler.step()
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

final_report = {
    "seed": SEED,
    "split_seed": SPLIT_SEED,
    "device": str(DEVICE),
    "best_auc": float(best_auc),
    "split": split_meta,
}
try:
    final_report.update({
        "last_val": {
            "loss": float(val_loss),
            "acc": float(val_m["acc"]),
            "pos_recall": float(val_m["pos_recall"]),
            "neg_recall": float(val_m["neg_recall"]),
            "auc": float(val_m["auc"]),
            "auprc": float(val_m["auprc"]),
            "margin_mu": float(val_m["mu"]),
            "margin_sd": float(val_m["sd"]),
        }
    })
except Exception:
    pass

with open(os.path.join(SAVE_DIR, "run.json"), "w") as f:
    json.dump(final_report, f, indent=2)

print(f"[SAVED] {ckpt_path} best_auc={best_auc:.4f} split_mode={SPLIT_MODE}")
