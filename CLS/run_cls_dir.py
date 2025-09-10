import os, json, pickle, time
import torch, torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, average_precision_score
from torch.utils.data import DataLoader, random_split

from loader import ProteinDrugInteractionDataset, collate_fn
from model import ExpressionDirectionClassifier  # <-- put your model in this module/file
from gnn import GNN_graphpred

# ---------- CONFIG ----------
CSV_PATH    = os.getenv("CSV_PATH")                      # e.g., /.../expr_dir.csv
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 4))
EPOCHS      = int(os.getenv("EPOCHS", 50))
LR          = float(os.getenv("LR", 3e-4))
WEIGHT_DECAY= float(os.getenv("WEIGHT_DECAY", 0.01))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LMDB_PATH   = os.getenv("LMDB_PATH",   "/mnt/data/gene_data/lmdb_parts")
DRUG_CACHE  = os.getenv("DRUG_CACHE",  "/mnt/data/graph_cache.pkl")
GID2LMDB_JSON = os.getenv("GID2LMDB_JSON", "/mnt/data/geneid_to_lmdb.json")

USE_XATTN   = os.getenv("USE_XATTN", "true").lower() == "true"
FREEZE_GNN  = os.getenv("FREEZE_GNN","true").lower() == "true"
GNN         = os.getenv("GNN", "gin")                    # 'gcn' or 'gin'
GNN_CKPT    = '/mnt/data/gcn_contextpred.pth' if GNN=="gcn" else '/mnt/data/gin_contextpred.pth'

DIM_P       = int(os.getenv("DIM_P", 768))               # gene/DNA embedding dim
EMB_DIM     = int(os.getenv("EMB_DIM", 300))             # drug GNN hidden dim
D_MODEL     = int(os.getenv("D_MODEL", 512))
N_CLASSES   = 2                                          # binary direction


# ---------- EVAL (direction only) ----------
def eval_reports_direction(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for b in loader:
            p = b["protein_embeddings"].to(device)  # [B, DIM_P]
            d = b["drug_graphs"].to(device)         # PyG Batch
            y = b["direction_ids"].to(device)       # {0,1}
            ld = model(p, d)                        # [B,2] logits
            prob_pos = ld.softmax(-1)[:, 1]
            y_true.extend(y.cpu().tolist())
            y_prob.extend(prob_pos.cpu().tolist())
    y_true = np.array(y_true)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    rpt = classification_report(y_true, y_pred, target_names=["neg","pos"], zero_division=0)
    return pr_auc, rpt


# ---------- MAIN ----------
def main():
    df = pd.read_csv(CSV_PATH)
    with open(DRUG_CACHE, "rb") as f:
        graph_cache = pickle.load(f)

    full_set = ProteinDrugInteractionDataset(
        df, LMDB_PATH, graph_cache, gid2lmdb_json=GID2LMDB_JSON
    )

    # 80/10/10 random split
    n_total = len(full_set)
    n_train = int(0.8 * n_total)
    n_val   = int(0.1 * n_total)
    n_test  = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(full_set, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    gnn = GNN_graphpred(5, EMB_DIM, EMB_DIM, graph_pooling=None, gnn_type='gin')
    model = ExpressionDirectionClassifier(
        dim_p=DIM_P, emb_dim=EMB_DIM, d_model=D_MODEL,
        gnn=gnn,
        gnn_ckpt=GNN_CKPT, freeze_gnn=FREEZE_GNN, use_xattn=USE_XATTN, n_classes=N_CLASSES
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    print(f"[run] n={n_total} | train={n_train} val={n_val} test={n_test} | lr={LR} | xattn={USE_XATTN} | gnn={GNN}")

    best = -1.0; best_state = None
    for ep in range(1, EPOCHS+1):
        model.train(); total = 0.0
        for b in train_loader:
            p = b["dna_embeddings"].to(DEVICE)   # [B, T, dim]
            pm = b["dna_mask"].to(DEVICE)        # [B, T] (True=PAD)
            d = b["drug_graphs"].to(DEVICE)

            y = b["direction_ids"].to(DEVICE)       # {0,1}

            opt.zero_grad(set_to_none=True)
            ld = model(p, d, p_pad=pm)                         # [B,2]
            loss = crit(ld, y)
            loss.backward(); opt.step()
            total += loss.item()

        pr_auc, rpt = eval_reports_direction(model, val_loader, DEVICE)
        print(f"[E{ep}] loss={total/len(train_loader):.4f} | val_PR-AUC={pr_auc:.4f}")
        if pr_auc > best:
            best = pr_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # test
    if best_state: model.load_state_dict(best_state)
    pr_auc, rpt = eval_reports_direction(model, test_loader, DEVICE)
    print("\n[TEST] PR-AUC:", f"{pr_auc:.4f}")
    print("[TEST] Report:\n", rpt)


if __name__ == "__main__":
    main()
