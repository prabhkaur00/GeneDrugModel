# DrugGene: Multimodal Drug–Gene Interaction Classification

This repository provides a training pipeline for predicting **drug–gene interaction directions** (e.g., *increase* vs. *decrease* expression).  
The model combines **mean-pooled gene embeddings** with **attention-pooled drug graph embeddings** for multimodal classification.

---

## from usageFeatures
- **Multimodal input**
  - Gene sequences → precomputed **mean-pooled embeddings** (LMDB).
  - Drug molecules → **graph neural networks (GIN/GCN)** with **attention pooling**.
- **Configurable GNN backbone**: freeze or fine-tune pretrained weights.
- **Optuna-based hyperparameter tuning** for LR, weight decay, batch size, GNN type, etc.
- **Deterministic training** via seeding for reproducibility.

---

## Dataset
- **Genes**: stored as LMDB, each entry is a mean-pooled embedding vector.
- **Drugs**: cached PyTorch Geometric graphs (SMILES → molecular graph → attention-pooled embedding).
- **CSV split file**: lists drug–gene pairs with interaction **direction labels**.
Data can be found here - https://drive.google.com/drive/folders/17JQhsiVpkugNg1W_R7rcsvdPFtGtETqI?usp=drive_link


## Usage

### Train a model
```bash
EPOCHS=20 BATCH_SIZE=128 LR=3e-4 WEIGHT_DECAY=0.01 \
CSV_PATH=/path/to/direction_cls.csv \
LMDB_PATH=/path/to/mean-pooled-all.lmdb \
DRUG_CACHE=/path/to/smiles_cache.pkl \
SAVE_DIR=./runs \
python train_direction.py
```

### Hyperparameter tuning with Optuna
```bash
OPTUNA_N_TRIALS=30 EPOCHS=10 \
CSV_PATH=/path/to/direction_cls.csv \
LMDB_PATH=/path/to/mean-pooled-all.lmdb \
DRUG_CACHE=/path/to/smiles_cache.pkl \
SAVE_DIR=./runs \
python train_direction.py
```

- Runs `30` trials, each for `10` epochs.  
- Best trial results are saved to `optuna_best.json`.  

---

## Outputs
- `runs/model.pt` – trained model weights  
- `runs/run.json` – config & metrics of the run  
- `runs/optuna_best.json` – best Optuna trial parameters  


