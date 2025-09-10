# train_direction.py
import os, json, time, pickle, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset

from loader_simple import ProteinDrugInteractionDataset, collate_fn
from model import ExpressionDirectionClassifier  # from your refactor
from sklearn.metrics import roc_auc_score, average_precision_score


CSV_PATH    = os.getenv("CSV_PATH", "/mnt/data/5k/simple-cls/direction_cls.csv")
LMDB_PATH   = os.getenv("LMDB_PATH", "/mnt/data/gene_data/mean-pooled-all.lmdb")
DRUG_CACHE  = os.getenv("DRUG_CACHE", "/mnt/data/graph_cache.pkl")
GNN         = os.getenv("GNN", "gin")
USE_XATTN   = os.getenv("USE_XATTN", "false").lower() == "true"
FREEZE_GNN  = os.getenv("FREEZE_GNN", "true").lower() == "true"
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 128))
EPOCHS      = int(os.getenv("EPOCHS", 20))
LR          = float(os.getenv("LR", 3e-4))
WEIGHT_DECAY= float(os.getenv("WEIGHT_DECAY", 0.01))
VAL_FRACTION= float(os.getenv("VAL_FRACTION", 0.2))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
SAVE_DIR    = os.getenv("SAVE_DIR", "/mnt/data/cls/runs/direction")
SEED        = int(os.getenv("SEED", 42))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn_ckpt    = '/mnt/data/gcn_contextpred.pth' if GNN=="gcn" else '/mnt/data/gin_contextpred.pth'

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)
df = df[(df["target_id"] == 0) & (df["direction_id"].isin([0,1]))].reset_index(drop=True)

with open(DRUG_CACHE, "rb") as f:
    graph_cache = pickle.load(f)

dataset = ProteinDrugInteractionDataset(
    df, LMDB_PATH, graph_cache,
)

idx = np.arange(len(dataset))
np.random.shuffle(idx)
n_val = int(len(idx)*VAL_FRACTION)
val_idx = idx[:n_val] if n_val>0 else []
train_idx = idx[n_val:] if n_val>0 else idx

train_set = Subset(dataset, train_idx)
val_set   = Subset(dataset, val_idx) if n_val>0 else None

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
# sanity check one batch
b = next(iter(train_loader))
assert b["protein_embeddings"].ndim == 2 and b["protein_embeddings"].shape[1] == 768
print("batch protein shape:", tuple(b["protein_embeddings"].shape))
print("batch drug graphs:", b["drug_graphs"])
print("batch labels (dir):", b["direction_ids"].unique(sorted=True))

val_loader = None if val_set is None else DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

model = ExpressionDirectionClassifier(
    dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=gnn_ckpt,
    freeze_gnn=FREEZE_GNN, gnn_type=GNN, use_xattn=USE_XATTN, n_classes=2
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
crit = nn.CrossEntropyLoss()

run_cfg = {
    "task":"direction_cls",
    "csv":CSV_PATH,
    "samples_total":len(dataset),
    "train":len(train_set),
    "val":len(val_set) if val_set else 0,
    "batch_size":BATCH_SIZE,
    "epochs":EPOCHS,
    "lr":LR,
    "weight_decay":WEIGHT_DECAY,
    "use_xattn":USE_XATTN,
    "freeze_gnn":FREEZE_GNN,
    "gnn":GNN,
    "seed":SEED,
    "device":str(DEVICE)
}
print("[CONFIG]", json.dumps(run_cfg, indent=2))
best_val = float("inf"); best_state=None
for ep in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    tot=0.0; nb=0
    tr_probs=[]; tr_tgts=[]
    for b in train_loader:
        p = b["protein_embeddings"].to(DEVICE)
        d = b["drug_graphs"].to(DEVICE)
        y = b["direction_ids"].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(p, d)
        loss = crit(logits, y)
        loss.backward(); opt.step()
        tot += loss.item(); nb += 1
        tr_probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())
        tr_tgts.append(y.detach().cpu())
    train_loss = tot/max(1,nb)
    yt = torch.cat(tr_tgts).numpy()
    pt = torch.cat(tr_probs).numpy()
    try:
        train_auroc = roc_auc_score(yt, pt)
    except Exception:
        train_auroc = float("nan")
    try:
        train_auprc = average_precision_score(yt, pt)
    except Exception:
        train_auprc = float("nan")

    if val_loader is not None:
        model.eval()
        vtot=0.0; vnb=0
        vl_probs=[]; vl_tgts=[]
        with torch.no_grad():
            for b in val_loader:
                p = b["protein_embeddings"].to(DEVICE)
                d = b["drug_graphs"].to(DEVICE)
                y = b["direction_ids"].to(DEVICE)
                logits = model(p, d)
                loss = crit(logits, y)
                vtot += loss.item(); vnb += 1
                vl_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
                vl_tgts.append(y.cpu())
        val_loss = vtot/max(1,vnb)
        yv = torch.cat(vl_tgts).numpy()
        pv = torch.cat(vl_probs).numpy()
        try:
            val_auroc = roc_auc_score(yv, pv)
        except Exception:
            val_auroc = float("nan")
        try:
            val_auprc = average_precision_score(yv, pv)
        except Exception:
            val_auprc = float("nan")
    else:
        val_loss = train_loss
        val_auroc = float("nan")
        val_auprc = float("nan")

    dt = time.time()-t0
    print(f"[E{ep}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"train_auroc={train_auroc:.4f} train_auprc={train_auprc:.4f} "
          f"val_auroc={val_auroc:.4f} val_auprc={val_auprc:.4f} time_s={dt:.1f}")

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

ckpt_path = os.path.join(SAVE_DIR, "model.pt")
torch.save(model.state_dict(), ckpt_path)

with open(os.path.join(SAVE_DIR, "run.json"), "w") as f:
    json.dump({**run_cfg, "best_val_loss": float(best_val)}, f, indent=2)

print("[SAVED]", ckpt_path)
