# DrugGene: Multimodal Drug–Gene Interaction Classification

This repository provides a reproducible training pipeline for predicting **drug–gene interaction directions** — such as whether a drug *increases* or *decreases* the expression of a target gene.

We explore two key modeling setups:

1. **Bilinear-fusion model** using **mean-pooled gene and drug embeddings**, with experiments comparing **GCN vs GIN** for drug encoders (inspired by [DrugChat](#citations)). Transitioning from GCN to GIN and unfreezing encoders improved validation loss by approximately 5%.

2. **Cross-attention fusion model** using **DNABERT-filtered gene sequences** (≤500 and ≤1000 tokens) combined with drug graph embeddings. AUC and AUPRC peaked around **0.70** — see plots below.

---

## Contributions

- **Dataset Construction**  
  Scraped and integrated over **2.5 million drug–gene interactions** from public sources including CTD, PubChem, and NCBI. Each entry includes:
  - SMILES representation (drug structure)
  - Gene ID and DNA sequence
  - Description of interaction
  - Labels for interaction direction (increase, decrease, none) and type (e.g., expression, methylation)

- **Expression Modulation Task**  
  This work focuses on predicting **how** a drug modulates gene expression, which is a less commonly explored task than binary drug–target interaction (DTI) prediction. The dataset is filtered specifically for this direction-label prediction task.

---

## Model Overview

- **Multimodal Inputs**
  - **Gene Sequences** → Mean-pooled embeddings from DNABERT (precomputed and provided for ≤1000 token sequences). Full-sequence embeddings can be reproduced using the DNABERT model.
  - **Drug Molecules** → Graph neural networks (**GIN** or **GCN**) trained over molecular graphs derived from SMILES, with attention pooling.

- **Architecture Configurability**
  - Easily switch GNN backbone
  - Choose between frozen or fine-tuned encoder components

- **Reproducible Training**
  - Deterministic seeding across DataLoaders, model weights, and torch operations

---

## Dataset Format

- **Genes**: Stored in LMDB format, indexed by gene ID
- **Drugs**: Preprocessed into PyTorch Geometric `Data` objects from SMILES strings and cached
- **Pairs**: A CSV file defines training/validation/test splits with corresponding interaction direction labels

Data can be accessed from:
https://drive.google.com/drive/folders/17JQhsiVpkugNg1W_R7rcsvdPFtGtETqI?usp=drive_link


---

## Results

### AUC and AUPRC across Epochs (500 vs 1000-token Gene Sequences)

![AUC vs AUPRC](./auc_auprc_plot.png)

- AUC and AUPRC both improve steadily for the first 7–8 epochs and then plateau
- Performance for 1K-token inputs slightly exceeds that of 500-token inputs
- Final AUC ~0.693–0.695, indicating meaningful signal capture beyond class imbalance baseline

---

## Citations

This project builds on the following foundational works for gene and molecule representations:

- DNABERT:  
  Ji, Y., Zhou, Z., Liu, H., & Davuluri, R. V. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*, 37(15), 2112–2120.

- DrugChat:  
  Jiang, M. et al. (2023). DrugChat: Towards Drug Discovery with Fine-Tuned Language Models on Biomedical Knowledge Graph. *arXiv preprint arXiv:2305.14326*.
