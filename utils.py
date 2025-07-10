import torch
from torch_geometric.data import Data
import gc
from torch_geometric.data import Batch

def cache_to_pyg_data(graph_dict):
    """Convert cached graph to PyG Data object"""
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
    edge_feat = torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
    node_feat = torch.tensor(graph_dict['node_feat'], dtype=torch.long)
    num_nodes = graph_dict['num_nodes']

    data = Data(
        x=node_feat,
        edge_index=edge_index,
        edge_attr=edge_feat,
        num_nodes=num_nodes
    )

    return data


def optimize_memory():
    """Clear memory to prevent OOM"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
if torch.cuda.is_available():
    device = torch.device("cuda")
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    print(f"GPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved:  {reserved:.2f} GB")
else:
    print("CUDA not available.")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def collate_fn(batch):
    """Fixed collate function with proper device handling"""
    max_protein_len = max([item['protein_embedding'].shape[0] for item in batch])

    padded_proteins = []
    for item in batch:
        protein_emb = item['protein_embedding']
        current_len = protein_emb.shape[0]

        if current_len < max_protein_len:
            padding = torch.zeros(max_protein_len - current_len, protein_emb.shape[1], dtype=protein_emb.dtype)
            padded_protein = torch.cat([protein_emb, padding], dim=0)
        else:
            padded_protein = protein_emb[:max_protein_len]
        padded_proteins.append(padded_protein)

    protein_embeddings = torch.stack(padded_proteins)
    drug_graphs = [item['drug_graph'] for item in batch]
    interactions = [item['interaction'] for item in batch]
    gene_ids = [item['gene_id'] for item in batch]
    smiles = [item['smiles'] for item in batch]
    batched_drugs = Batch.from_data_list(drug_graphs)
    encoded_texts = None
    if all(item['encoded_text'] is not None for item in batch):
        encoded_texts = {
            k: torch.stack([item['encoded_text'][k] for item in batch])
            for k in batch[0]['encoded_text'].keys()
        }

    return {
        'protein_embeddings': protein_embeddings,
        'drug_graphs': batched_drugs,
        'interactions': interactions,
        'encoded_texts': encoded_texts,
        'gene_ids': gene_ids,
        'smiles': smiles
    }

def count_parameters(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params
