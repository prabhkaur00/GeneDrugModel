import os, json, time, pickle, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import multiprocessing as mp
from loader_xattn import ProteinDrugInteractionDataset, collate_fn
from model_xattn import ExpressionDirectionClassifier
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score

# Cluster-safe multiprocessing and memory settings
torch.multiprocessing.set_start_method('spawn', force=True)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))  # Force 0 to avoid segfaults
SAVE_DIR    = os.getenv("SAVE_DIR", "/mnt/data/cls/runs/direction")
SEED        = int(os.getenv("SEED", 42))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn_ckpt    = '/mnt/data/gcn_contextpred.pth' if GNN=="gcn" else '/mnt/data/gin_contextpred.pth'
RETURN_SEQS = os.getenv("RETURN_SEQUENCES", "true").lower() == "true"
GENE_POOL_F = int(os.getenv("GENE_POOL_FACTOR", 4))
MAX_NODES   = int(os.getenv("MAX_NODES", 50))

torch.set_num_threads(1)
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)
print(f"[RAW_CSV] Total rows: {len(df)}")
print(f"[RAW_CSV] Columns: {list(df.columns)}")
print(f"[RAW_CSV] target_id unique: {df['target_id'].unique()[:10]}")
print(f"[RAW_CSV] direction_id unique: {df['direction_id'].unique()}")
print(f"[RAW_CSV] direction_id counts before filter: {Counter(df['direction_id'].tolist())}")

df = df[(df["target_id"] == 0) & (df["direction_id"].isin([0,1]))].reset_index(drop=True)
ctr = Counter(df["direction_id"].tolist())
print(f"[FILTERED_CSV] N={len(df)} class_counts={dict(ctr)} pos_ratio={ctr.get(1,0)/max(1,sum(ctr.values())):.4f}")

# Critical check: verify we have both classes after filtering
if len(ctr) == 1:
    print("CRITICAL: Only one class remains after filtering! Check CSV filtering logic.")
if ctr.get(1, 0) == 0:
    print("CRITICAL: No positive samples (direction_id=1) found after filtering!")

# Class weight calculation with validation
total_samples = len(df)
class_0_count = ctr.get(0, 0)
class_1_count = ctr.get(1, 0)

if class_0_count == 0 or class_1_count == 0:
    print("CRITICAL: Missing one or both classes! Using equal weights.")
    weight_0 = weight_1 = 1.0
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)
else:
    weight_0 = total_samples / (2 * class_0_count)
    weight_1 = total_samples / (2 * class_1_count)
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(DEVICE)

print(f"[WEIGHTS] class_weights=[{weight_0:.3f}, {weight_1:.3f}] class_counts=[{class_0_count}, {class_1_count}]")

with open(DRUG_CACHE, "rb") as f:
    graph_cache = pickle.load(f)

dataset = ProteinDrugInteractionDataset(
    seg_df=df,
    lmdb_path=LMDB_DIR,
    smiles_cache=graph_cache,
    verbose=False  # Reduce noise
)

# Comprehensive dataset validation
print("[VALIDATION] Checking first 10 samples and data loading...")
sample_labels = []
for i in range(min(10, len(dataset))):
    try:
        sample = dataset[i]
        label = sample["direction_id"]
        sample_labels.append(label.item() if hasattr(label, 'item') else label)
        print(f"Sample {i}: label={label} dtype={type(label)} tensor_dtype={label.dtype if hasattr(label, 'dtype') else 'N/A'}")
        
        if torch.isnan(sample["protein"]).any():
            print(f"WARNING: NaN in protein_embeddings at sample {i}")
        if torch.isinf(sample["protein"]).any():
            print(f"WARNING: Inf in protein_embeddings at sample {i}")
    except Exception as e:
        print(f"ERROR: Cannot load sample {i}: {e}")

print(f"[SAMPLE_LABELS] First 10 labels: {sample_labels}")
print(f"[SAMPLE_DISTRIBUTION] {Counter(sample_labels)}")

idx = np.arange(len(dataset))
np.random.shuffle(idx)
n_val = int(len(idx)*VAL_FRACTION)
val_idx = idx[:n_val] if n_val>0 else []
train_idx = idx[n_val:] if n_val>0 else idx

# Check train/val split preserves class distribution
train_labels_from_df = [df.iloc[idx]["direction_id"] for idx in train_idx[:100]]  # Check first 100
val_labels_from_df = [df.iloc[idx]["direction_id"] for idx in val_idx[:100]] if len(val_idx) > 0 else []
print(f"[SPLIT_CHECK] Train first 100: {Counter(train_labels_from_df)}")
print(f"[SPLIT_CHECK] Val first 100: {Counter(val_labels_from_df)}")

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx) if n_val>0 else None

# Safer DataLoader configuration
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Force 0 workers to prevent segfaults
    pin_memory=False,
    collate_fn=collate_fn,
    drop_last=True,  # Avoid incomplete batches
)

val_loader = None if val_set is None else DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_fn,
    drop_last=False,
)

print(f"[TRAIN] batch_size={BATCH_SIZE} steps_per_epoch={len(train_loader)} workers=0")

# Critical: Check actual data distribution in loaders
print("[LOADER_CHECK] Analyzing first training batch...")
_check_loader = DataLoader(train_set, batch_size=min(32, len(train_set)), shuffle=False, num_workers=0, collate_fn=collate_fn)
_check_batch = next(iter(_check_loader))
_check_labels = _check_batch["direction_ids"]
_check_counter = Counter(_check_labels.numpy() if hasattr(_check_labels, 'numpy') else _check_labels.tolist())
print(f"[BATCH_LABELS] shape={_check_labels.shape} dtype={_check_labels.dtype}")
print(f"[BATCH_DISTRIBUTION] {dict(_check_counter)} total={len(_check_labels)}")
print(f"[BATCH_UNIQUE] unique_values={torch.unique(_check_labels).tolist()}")

# Check if labels are being corrupted during dataset loading vs CSV
print("[DATASET_VS_CSV] Comparing first 5 samples...")
for i in range(min(5, len(dataset))):
    csv_label = df.iloc[i]["direction_id"]
    dataset_sample = dataset[i]
    dataset_label = dataset_sample["direction_id"]
    print(f"Sample {i}: CSV={csv_label} Dataset={dataset_label} Match={csv_label==dataset_label}")

if len(_check_counter) == 1:
    print("CRITICAL: Only one class found in batch! Checking individual samples...")
    for i in range(min(5, len(train_set))):
        sample = train_set[i]
        original_idx = train_idx[i]
        csv_label = df.iloc[original_idx]["direction_id"]
        dataset_label = sample['direction_id']
        print(f"TrainSet[{i}] original_idx={original_idx} CSV={csv_label} Dataset={dataset_label}")

del _check_loader, _check_batch, _check_labels

model = ExpressionDirectionClassifier(
    dim_p=768,
    emb_dim=300,
    d_model=512,
    gnn_ckpt=gnn_ckpt,
    freeze_gnn=FREEZE_GNN,
    gnn_type=GNN,
    use_xattn=USE_XATTN,
    n_classes=2,
    return_sequences=RETURN_SEQS,
).to(DEVICE)

# Differential learning rates
head_params = []
backbone_params = []
for name, param in model.named_parameters():
    if 'head' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

opt = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LR * 0.1},
    {'params': head_params, 'lr': LR}
], weight_decay=WEIGHT_DECAY)

# Weighted loss for class imbalance
crit = nn.CrossEntropyLoss(weight=class_weights)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=5, verbose=True
)

def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def epoch_metrics(all_logits, all_y):
    if not all_logits:
        return dict(acc=float("nan"), pos_recall=float("nan"), neg_recall=float("nan"),
                    mu=float("nan"), sd=float("nan"), auc=float("nan"), auprc=float("nan"))
    y = torch.cat(all_y).detach().cpu().long()
    logits = torch.cat(all_logits).detach().cpu()
    probs = logits.softmax(dim=-1)[:, 1]
    preds = logits.argmax(dim=-1)

    acc = (preds == y).float().mean().item()
    c = Counter(y.tolist())
    pos = c.get(1, 0); neg = c.get(0, 0)
    pos_recall = ((preds[y == 1] == 1).float().mean().item()) if pos > 0 else float('nan')
    neg_recall = ((preds[y == 0] == 0).float().mean().item()) if neg > 0 else float('nan')
    mu, sd = logits.mean().item(), logits.std().item()
    try:
        auc = roc_auc_score(y.numpy(), probs.numpy()) if len(set(y.numpy())) > 1 else float('nan')
        auprc = average_precision_score(y.numpy(), probs.numpy()) if len(set(y.numpy())) > 1 else float('nan')
    except Exception:
        auc, auprc = float('nan'), float('nan')
    return dict(acc=acc, pos_recall=pos_recall, neg_recall=neg_recall, mu=mu, sd=sd, auc=auc, auprc=auprc)

best_val = float("inf"); best_state=None
early_stop_patience = 10
patience_counter = 0

for ep in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    tot=0.0; nb=0
    all_logits_t, all_y_t = [], []
    
    try:
        for batch_idx, b in enumerate(train_loader):
            try:
                p = b["protein_embeddings"].to(DEVICE, non_blocking=True)
                pm = b["protein_pad_mask"].to(DEVICE, non_blocking=True)
                y = b["direction_ids"].to(DEVICE, non_blocking=True).long()

                # Debug: Check batch label distribution for first few epochs
                if ep <= 3 and batch_idx == 0:
                    batch_counter = Counter(y.cpu().numpy())
                    print(f"[E{ep}_B{batch_idx}] batch_labels={dict(batch_counter)} unique={torch.unique(y).tolist()}")

                opt.zero_grad(set_to_none=True)
                logits = model(p_vec=p, batch_data=b["drug_graphs"], p_mask=pm)

                loss = crit(logits, y)
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Invalid loss at batch {batch_idx}: {loss.item()}")
                    continue

                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                opt.step()

                tot += loss.item(); nb += 1
                all_logits_t.append(logits.detach())
                all_y_t.append(y.detach())
                
                # Periodic memory cleanup
                if batch_idx % 50 == 0:
                    cleanup_memory()
                    
            except RuntimeError as e:
                print(f"ERROR in batch {batch_idx}: {e}")
                cleanup_memory()
                continue
                
    except Exception as e:
        print(f"CRITICAL ERROR in epoch {ep}: {e}")
        cleanup_memory()
        break

    train_loss = tot/max(1,nb)
    train_m = epoch_metrics(all_logits_t, all_y_t)
    
    # Debug: Log detailed class distribution for first few epochs
    if ep <= 5:
        all_train_y = torch.cat(all_y_t).cpu()
        train_dist = Counter(all_train_y.numpy())
        print(f"[E{ep}_TRAIN_DIST] {dict(train_dist)} total={len(all_train_y)}")

    if val_loader is not None:
        model.eval()
        vtot=0.0; vnb=0
        all_logits_v, all_y_v = [], []
        with torch.no_grad():
            for b in val_loader:
                p = b["protein_embeddings"].to(DEVICE, non_blocking=True)
                pm = b["protein_pad_mask"].to(DEVICE, non_blocking=True)
                y = b["direction_ids"].to(DEVICE, non_blocking=True).long()
                logits = model(p_vec=p, batch_data=b["drug_graphs"], p_mask=pm)

                loss = crit(logits, y)
                vtot += loss.item(); vnb += 1
                all_logits_v.append(logits)
                all_y_v.append(y)
        val_loss = vtot/max(1,vnb)
        val_m = epoch_metrics(all_logits_v, all_y_v)
    else:
        val_loss = train_loss
        val_m = train_m

    # Update scheduler
    scheduler.step(val_loss)

    dt = time.time()-t0
    print(
        f"[E{ep}] "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_acc={train_m['acc']:.3f} val_acc={val_m['acc']:.3f} "
        f"posR={train_m['pos_recall']:.3f}/{val_m['pos_recall']:.3f} "
        f"negR={train_m['neg_recall']:.3f}/{val_m['neg_recall']:.3f} "
        f"val_logits_mu,sd={val_m['mu']:.3f},{val_m['sd']:.3f} "
        f"AUC/AUPRC={val_m['auc']:.3f}/{val_m['auprc']:.3f} "
        f"time_s={dt:.1f}")

    # Log gradients only for key epochs
    if ep in (1, 2, 5) or ep % 10 == 0:
        with torch.no_grad():
            gnorms = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    gnorms.append((name, p.grad.norm().item()))
            gnorms = sorted(gnorms, key=lambda x: -x[1])[:8]
            print("[GRADS top8]", gnorms)

    # Early stopping logic
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        
    # Stop if model not learning effectively
    if val_m['auc'] < 0.55 and ep > 15:
        print(f"[EARLY STOP] AUC not improving after {ep} epochs")
        break
        
    if patience_counter >= early_stop_patience:
        print(f"[EARLY STOP] No improvement for {early_stop_patience} epochs")
        break

if best_state is not None:
    model.load_state_dict(best_state)

ckpt_path = os.path.join(SAVE_DIR, "model.pt")
torch.save(model.state_dict(), ckpt_path)

with open(os.path.join(SAVE_DIR, "run.json"), "w") as f:
    json.dump({"seed": SEED, "device": str(DEVICE), "best_val_loss": float(best_val)}, f, indent=2)

print(f"[SAVED] {ckpt_path} best_val={best_val:.4f}")