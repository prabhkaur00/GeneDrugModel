import os, json, pickle, datetime
import torch, torch.nn as nn
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, average_precision_score, brier_score_loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import StageBJoint
from loader import ProteinDrugInteractionDataset, collate_fn
import numpy as np, time

# ---------- CONFIG ----------
CSV_PATH    = os.getenv("CSV_PATH")  # e.g., /.../segments_stageB_expr_meth.csv
SPLIT_DIR   = os.getenv("SPLIT_DIR") # e.g., 5k/splits
SPLIT_MODE  = os.getenv("SPLIT_MODE", "random")  # random | cold_both | cold_gene | cold_drug
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 128))
EPOCHS      = int(os.getenv("EPOCHS", 8))
LR          = float(os.getenv("LR", 3e-4))
WEIGHT_DECAY= float(os.getenv("WEIGHT_DECAY", 0.01))
IMB_STRAT   = os.getenv("IMBALANCE_STRATEGY", "focal")  # focal|weights|sampler|none
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", 1.75))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LMDB_PATH   = os.getenv("LMDB_PATH",   "/mnt/data/gene_data/lmdb_parts")
DRUG_CACHE  = os.getenv("DRUG_CACHE",  "/mnt/data/graph_cache.pkl")
VOCAB_PATH  = os.getenv("VOCAB_PATH",  "/mnt/data/cls/vocab_minimal.json")
USE_XATTN   = os.getenv("USE_XATTN", "true").lower() == "true"
FREEZE_GNN  = os.getenv("FREEZE_GNN","true").lower() == "true"
GNN         = os.getenv("GNN", "gin")
gnn_ckpt    = '/mnt/data/gcn_contextpred.pth' if GNN=="gcn" else '/mnt/data/gin_contextpred.pth'



def recall_at_precision(y_true, y_score, target_precision=0.8):
    # y_score: prob for methylation class (1)
    order = np.argsort(-y_score)
    tp = fp = 0
    pos = (np.array(y_true) == 1).sum()
    best = 0.0
    for i in order:
        if y_true[i] == 1: tp += 1
        else: fp += 1
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, pos)
        if prec >= target_precision:
            best = max(best, rec)
    return best

def expected_calibration_error(y_true, y_prob, n_bins=10):
    # simple ECE for binary methylation (class=1)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for a,b in zip(bins[:-1], bins[1:]):
        m = (y_prob >= a) & (y_prob < b)
        if m.any():
            conf = y_prob[m].mean()
            acc  = (y_true[m] == (y_prob[m] >= 0.5)).mean()
            ece += (m.mean()) * abs(acc - conf)
    return ece
# ---------- LOSS ----------
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
    raise ValueError(kind)

# ---------- SPLIT LOADER ----------
def load_split_dfs(df, split_dir, mode):
    if mode == "random":
        idx = json.load(open(os.path.join(split_dir, "random_idx.json")))
        tr = df.iloc[idx["train"]].reset_index(drop=True)
        va = df.iloc[idx["val"]].reset_index(drop=True)
        te = df.iloc[idx["test"]].reset_index(drop=True)
        return tr, va, te
    if mode == "cold_both":
        idx = json.load(open(os.path.join(split_dir, "cold_both_idx.json")))
        tr = df.iloc[idx["train"]].reset_index(drop=True)
        va = df.iloc[idx["val"]].reset_index(drop=True)
        te = df.iloc[idx["test"]].reset_index(drop=True)
        return tr, va, te
    if mode == "cold_gene":
        gids = json.load(open(os.path.join(split_dir, "cold_gene_ids.json")))
        tr = df[df["gene_id"].isin(set(gids["train"]))].reset_index(drop=True)
        va = df[df["gene_id"].isin(set(gids["val"]))].reset_index(drop=True)
        te = df[df["gene_id"].isin(set(gids["test"]))].reset_index(drop=True)
        return tr, va, te
    if mode == "cold_drug":
        dids = json.load(open(os.path.join(split_dir, "cold_drug_ids.json")))
        tr = df[df["smiles"].isin(set(dids["train"]))].reset_index(drop=True)
        va = df[df["smiles"].isin(set(dids["val"]))].reset_index(drop=True)
        te = df[df["smiles"].isin(set(dids["test"]))].reset_index(drop=True)
        return tr, va, te
    raise ValueError("SPLIT_MODE must be one of random|cold_both|cold_gene|cold_drug")

# ---------- EVAL ----------
def eval_reports(model, loader, device, target_names, dir_names):
    model.eval()
    yT, pT, yD, pD = [], [], [], []
    with torch.no_grad():
        for b in loader:
            p = b["protein_embeddings"].to(device)
            d = b["drug_graphs"].to(device)
            yt = b["target_ids"].to(device)
            yd = b["direction_ids"].to(device)
            lt, ld = model(p, d)
            yT.extend(yt.cpu().tolist()); yD.extend(yd.cpu().tolist())
            pT.extend(lt.softmax(-1)[:,1].cpu().tolist())  # methylation prob
            pD.extend(ld.softmax(-1).cpu().tolist())
    # PR-AUC for methylation (label id=1 from your mapping)
    yT_bin = [1 if v==1 else 0 for v in yT]
    pr_auc = average_precision_score(yT_bin, pT)
    rptT = classification_report(yT, [1 if x>=0.5 else 0 for x in pT], target_names=target_names, zero_division=0)
    # for direction we stick with argmax
    rptD = "skipped here for brevity"
    return pr_auc, rptT, rptD

# ---------- MAIN ----------
def main():
    "Starting run with splits"
    df = pd.read_csv(CSV_PATH)
    with open(VOCAB_PATH) as f:
        voc = json.load(f)
    target2id = voc["target2id"]; direction2id = voc["direction2id"]

    tr_df, va_df, te_df = load_split_dfs(df, SPLIT_DIR, SPLIT_MODE)

    with open(DRUG_CACHE, "rb") as f: graph_cache = pickle.load(f)

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
    # imbalance
    y_t = train_set.df["target_id"].to_numpy()
    y_d = train_set.df["direction_id"].to_numpy()

    from collections import Counter
    t_counts = Counter(y_t.tolist())
    d_counts = Counter(y_d.tolist())
    if IMB_STRAT == "weights":
        def cls_w(cnt, n): 
            import torch
            freq = torch.tensor([cnt.get(i,0) for i in range(n)], dtype=torch.float)
            freq = freq.clamp(min=1.0)
            w = torch.log(freq.sum()/freq)
            return (w / w.sum() * n).to(DEVICE)
        cw_t, cw_d = cls_w(t_counts, len(target2id)), cls_w(d_counts, len(direction2id))
        loss_kind = "cross_entropy"
        sampler = None
    elif IMB_STRAT == "sampler":
        from torch.utils.data import WeightedRandomSampler
        weights = [1.0/(t_counts[train_set[i]["target_id"]]+1e-6) for i in range(len(train_set))]
        sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)
        cw_t=cw_d=None; loss_kind="cross_entropy"
    elif IMB_STRAT == "focal":
        cw_t=cw_d=None; sampler=None; loss_kind="focal"
    else:
        cw_t=cw_d=None; sampler=None; loss_kind="cross_entropy"

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None),
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    model = StageBJoint(dim_p=768, emb_dim=300, d_model=512, n_dirs=len(direction2id),
                        gnn_ckpt=gnn_ckpt, freeze_gnn=FREEZE_GNN, gnn_type = GNN,use_xattn=USE_XATTN).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit_t = make_loss(loss_kind, weight=cw_t, gamma=FOCAL_GAMMA)
    crit_d = make_loss(loss_kind, weight=cw_d, gamma=FOCAL_GAMMA)

    print(f"[run] split={SPLIT_MODE} | train={len(train_set)} val={len(val_set)} test={len(test_set)} "
          f"| imb={IMB_STRAT} gamma={FOCAL_GAMMA} lr={LR}")

    best = -1.0; best_state = None
    for ep in range(1, EPOCHS+1):
        model.train(); total = 0.0
        n_batches = len(train_loader)
        print(f"[E{ep}] starting training with {n_batches} batches")

        for i, b in enumerate(train_loader, 1):
            p  = b["protein_embeddings"].to(DEVICE)
            d  = b["drug_graphs"].to(DEVICE)
            yt = b["target_ids"].to(DEVICE)
            yd = b["direction_ids"].to(DEVICE)

            opt.zero_grad(set_to_none=True)
            lt, ld = model(p, d)
            loss_t = crit_t(lt, yt)
            loss_d = crit_d(ld, yd)
            loss = loss_t + loss_d
            loss.backward(); opt.step()
            total += loss.item()

            # --- diagnostics ---
            n0 = (yt == 0).sum().item()
            n1 = (yt == 1).sum().item()
            if i % 50 == 0 or i == n_batches:
                print(f"[E{ep}] batch {i}/{n_batches} "
                    f"| target0={n0} target1={n1} "
                    f"| loss={loss.item():.4f}")

        # validation at end of epoch
        pr_auc, rptT, _ = eval_reports(model, val_loader, DEVICE,
                                    list(target2id.keys()), list(direction2id.keys()))
        print(f"[E{ep}] train_loss={total/n_batches:.4f} "
            f"| val_PR-AUC_methylation={pr_auc:.4f}")

        if pr_auc > best:
            best = pr_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # test with best
    if best_state: model.load_state_dict(best_state)
    pr_auc, rptT, _ = eval_reports(model, test_loader, DEVICE,
                                   list(target2id.keys()), list(direction2id.keys()))
    print("\n[TEST] PR-AUC (methylation):", f"{pr_auc:.4f}")
    print("[TEST] Target report:\n", rptT)

if __name__ == "__main__":
    main()