# train_direction.py
import os, json, time, pickle, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch

from loader_simple import ProteinDrugInteractionDataset, collate_fn
from model import ExpressionDirectionClassifier

CSV_PATH    = os.getenv("CSV_PATH", "/content/drive/MyDrive/DrugGeneModel-Data/direction_cls.csv")
LMDB_PATH   = os.getenv("LMDB_PATH", "/content/drive/MyDrive/DrugGeneModel-Data/mean-pooled-all.lmdb")
DRUG_CACHE  = os.getenv("DRUG_CACHE", "/content/drive/MyDrive/DrugGeneModel-Data/smiles_cache.pkl")
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
OPTUNA_N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", 0))
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def _ckpt_for(gnn_type):
    return '/content/drive/MyDrive/DrugGeneModel-Data/gcn_contextpred.pth' if gnn_type=="gcn" \
           else '/content/drive/MyDrive/DrugGeneModel-Data/gin_contextpred.pth'

df = pd.read_csv(CSV_PATH)
df = df[(df["target_id"] == 0) & (df["direction_id"].isin([0,1]))].reset_index(drop=True)

def cache_to_pyg_data(graph_dict):
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
    edge_feat = torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
    node_feat = torch.tensor(graph_dict['node_feat'], dtype=torch.long)
    num_nodes = graph_dict['num_nodes']
    return Data(x=node_feat, edge_index=edge_index, edge_attr=edge_feat, num_nodes=num_nodes)

with open(DRUG_CACHE, "rb") as f:
    smiles_cache_raw = pickle.load(f)
graph_cache = {k: cache_to_pyg_data(raw) for k, raw in smiles_cache_raw.items()}

dataset = ProteinDrugInteractionDataset(df, LMDB_PATH, graph_cache)
idx = np.arange(len(dataset)); np.random.shuffle(idx)
n_val = int(len(idx)*VAL_FRACTION)
val_idx = idx[:n_val] if n_val>0 else []
train_idx = idx[n_val:] if n_val>0 else idx
train_set = Subset(dataset, train_idx)
val_set   = Subset(dataset, val_idx) if n_val>0 else None

def build_loader(subset, bs, shuffle):
    return DataLoader(subset, batch_size=bs, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=True, collate_fn=collate_fn)

def run_once(lr, wd, bs, gnn_type, freeze_gnn, d_model):
    train_loader = build_loader(train_set, bs, True)
    b = next(iter(train_loader))
    assert b["protein_embeddings"].ndim == 2 and b["protein_embeddings"].shape[1] == 768
    val_loader = None if val_set is None else build_loader(val_set, bs, False)
    model = ExpressionDirectionClassifier(
        dim_p=768, emb_dim=300, d_model=d_model, gnn_ckpt=_ckpt_for(gnn_type),
        freeze_gnn=freeze_gnn, gnn_type=gnn_type, use_xattn=False, n_classes=2
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    best_val = float("inf"); best_state=None
    for ep in range(1, EPOCHS+1):
        model.train(); t0=time.time(); tot=0.0; nb=0
        for b in train_loader:
            p = b["protein_embeddings"].to(DEVICE)
            d = b["drug_graphs"].to(DEVICE)
            y = b["direction_ids"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(p,d), y); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        train_loss = tot/max(1,nb)
        if val_set is not None:
            model.eval(); vtot=0.0; vnb=0
            with torch.no_grad():
                for b in val_loader:
                    p = b["protein_embeddings"].to(DEVICE)
                    d = b["drug_graphs"].to(DEVICE)
                    y = b["direction_ids"].to(DEVICE)
                    vtot += crit(model(p,d), y).item(); vnb += 1
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
    return model, float(best_val)

def save_run(model, best_val, cfg, tag="model.pt"):
    ckpt_path = os.path.join(SAVE_DIR, tag)
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(SAVE_DIR, "run.json"), "w") as f:
        json.dump({**cfg, "best_val_loss": float(best_val)}, f, indent=2)
    print("[SAVED]", ckpt_path)

base_cfg = {
    "task":"direction_cls","csv":CSV_PATH,"samples_total":len(dataset),
    "train":len(train_set),"val":len(val_set) if val_set else 0,
    "epochs":EPOCHS,"seed":SEED,"device":str(DEVICE)
}

if OPTUNA_N_TRIALS <= 0:
    run_cfg = {
        **base_cfg,
        "batch_size":BATCH_SIZE,"lr":LR,"weight_decay":WEIGHT_DECAY,
        "use_xattn":USE_XATTN,"freeze_gnn":FREEZE_GNN,"gnn":GNN,"d_model":512
    }
    print("[CONFIG]", json.dumps(run_cfg, indent=2))
    model, best_val = run_once(LR, WEIGHT_DECAY, BATCH_SIZE, GNN, FREEZE_GNN, 512)
    save_run(model, best_val, run_cfg, "model.pt")
else:
    import optuna
    def objective(trial):
        gnn_type   = trial.suggest_categorical("gnn", ["gin","gcn"])
        freeze_gnn = trial.suggest_categorical("freeze_gnn", [True, False])
        d_model    = trial.suggest_categorical("d_model", [256, 384, 512, 640])
        lr         = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        wd         = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        bs         = trial.suggest_categorical("batch_size", [64, 128, 256])
        _, val = run_once(lr, wd, bs, gnn_type, freeze_gnn, d_model)
        return val
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
    best = study.best_trial
    print("[BEST]", best.number, best.value, best.params)
    bp = best.params
    run_cfg = {
        **base_cfg,
        "batch_size":bp["batch_size"],"lr":bp["lr"],"weight_decay":bp["weight_decay"],
        "freeze_gnn":bp["freeze_gnn"],
        "gnn":bp["gnn"],"d_model":bp["d_model"],"optuna_trials":OPTUNA_N_TRIALS
    }
    print("[CONFIG]", json.dumps(run_cfg, indent=2))
    model, best_val = run_once(bp["lr"], bp["weight_decay"], bp["batch_size"],
                               bp["gnn"], bp["freeze_gnn"], bp["d_model"])
    save_run(model, best_val, run_cfg, "model_best.pt")
    with open(os.path.join(SAVE_DIR, "optuna_best.json"), "w") as f:
        json.dump({"best_value": best.value, "best_params": best.params}, f, indent=2)
