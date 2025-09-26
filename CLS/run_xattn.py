# train_direction.py
import os, json, time, pickle, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import multiprocessing as mp
from loader_xattn import ProteinDrugInteractionDataset, collate_fn
from model_xattn import ExpressionDirectionClassifier  # from your refactor

CSV_PATH    = os.getenv("CSV_PATH", "/mnt/data/direction_lt500.csv")
LMDB_DIR   = os.getenv("LMDB_PATH", "/mnt/data/filtered_lt500.lmdb")
DRUG_CACHE  = os.getenv("DRUG_CACHE", "/mnt/data/graph_cache.pkl")
GNN         = os.getenv("GNN", "gin")
USE_XATTN   = os.getenv("USE_XATTN", "false").lower() == "true"
FREEZE_GNN  = os.getenv("FREEZE_GNN", "true").lower() == "true"
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 16))
EPOCHS      = int(os.getenv("EPOCHS", 100))
LR          = float(os.getenv("LR", 3e-4))
WEIGHT_DECAY= float(os.getenv("WEIGHT_DECAY", 0.01))
VAL_FRACTION= float(os.getenv("VAL_FRACTION", 0.2))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
SAVE_DIR    = os.getenv("SAVE_DIR", "/mnt/data/cls/runs/direction")
SEED        = int(os.getenv("SEED", 42))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn_ckpt    = '/mnt/data/gcn_contextpred.pth' if GNN=="gcn" else '/mnt/data/gin_contextpred.pth'
RETURN_SEQS = os.getenv("RETURN_SEQUENCES", "true").lower() == "true"   # use full sequences
GENE_POOL_F = int(os.getenv("GENE_POOL_FACTOR", 4))                      # 8× sliding window for gene
MAX_NODES   = int(os.getenv("MAX_NODES", 50))

torch.set_num_threads(1)
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)
df = df[(df["target_id"] == 0) & (df["direction_id"].isin([0,1]))].reset_index(drop=True)

with open(DRUG_CACHE, "rb") as f:
    graph_cache = pickle.load(f)

dataset = ProteinDrugInteractionDataset(
    seg_df=df,
    lmdb_path=LMDB_DIR,
    smiles_cache=graph_cache,
    verbose=True
)

idx = np.arange(len(dataset))
np.random.shuffle(idx)
n_val = int(len(idx)*VAL_FRACTION)
val_idx = idx[:n_val] if n_val>0 else []
train_idx = idx[n_val:] if n_val>0 else idx

train_set = Subset(dataset, train_idx)
val_set   = Subset(dataset, val_idx) if n_val>0 else None

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=(4 if NUM_WORKERS > 0 else None),
)

print(f"[TRAIN] batch_size={BATCH_SIZE} steps_per_epoch={len(train_loader)}")

# sanity check one batch
sanity_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
b = next(iter(sanity_loader))
pe = b["protein_embeddings"]
assert pe.shape[-1] == 768

val_loader = None if val_set is None else DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=(4 if NUM_WORKERS > 0 else None),
)
model = ExpressionDirectionClassifier(
    dim_p=768,
    emb_dim=300,
    d_model=512,
    gnn_ckpt=gnn_ckpt,
    freeze_gnn=FREEZE_GNN,
    gnn_type=GNN,
    use_xattn=USE_XATTN,
    n_classes=2,
    return_sequences=RETURN_SEQS,     # <- enable full sequences path
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
crit = nn.CrossEntropyLoss()

run_cfg = {
    "seed":SEED,
    "device":str(DEVICE)
}

best_val = float("inf"); best_state=None
for ep in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    tot=0.0; nb=0
    for b in train_loader:
        p   = b["protein_embeddings"].to(DEVICE)
        pm  = b["protein_pad_mask"].to(DEVICE)
        y   = b["direction_ids"].to(DEVICE)

        opt.zero_grad(set_to_none=True)
        logits = model(p_vec=p, batch_data=b["drug_graphs"], p_mask=pm)

        loss = crit(logits, y)
        loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    train_loss = tot/max(1,nb)

    if val_loader is not None:
        model.eval()
        vtot=0.0; vnb=0
        with torch.no_grad():
            for b in val_loader:
                p   = b["protein_embeddings"].to(DEVICE)
                pm  = b["protein_pad_mask"].to(DEVICE)
                y   = b["direction_ids"].to(DEVICE)

                logits = model(p_vec=p, batch_data=b["drug_graphs"], p_mask=pm)

                loss = crit(logits, y)
                vtot += loss.item(); vnb += 1
        val_loss = vtot/max(1,vnb)
    else:
        val_loss = train_loss

    dt = time.time()-t0
    print(f"[E{ep}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} time_s={dt:.1f}")

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
