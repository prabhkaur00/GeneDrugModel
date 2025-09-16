import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from gnn import GNN_graphpred

class WindowAttnPool(nn.Module):
    def __init__(self, d, factor=8, heads=1):
        super().__init__()
        self.factor = factor
        self.heads = heads
        self.q = nn.Parameter(torch.randn(heads, d))
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        W = self.factor
        pad = (-T) % W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        T2 = x.shape[1] // W
        x = x.view(B, T2, W, D)
        k = self.k(x)                              # [B,T2,W,D]
        v = self.v(x)                              # [B,T2,W,D]
        attn = torch.einsum('hd,btwd->bhtw', self.q, k) / (D ** 0.5)  # [B,H,T2,W]
        w = attn.softmax(-1)
        out = torch.einsum('bhtw,btwd->bhtd', w, v).mean(1)           # [B,T2,D]
        return out


class CrossAttnBlock(nn.Module):
    def __init__(self, d, nheads=4, ff=4):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff*d), nn.GELU(), nn.Linear(ff*d, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)
    def forward(self, p, d):
        x = torch.cat([p,d], dim=1)
        x = self.ln1(x + self.sa(x,x,x,need_weights=False)[0])
        p2 = self.ca(p, d, d, need_weights=False)[0]
        d2 = self.ca(d, p, p, need_weights=False)[0]
        x2 = torch.cat([p2,d2], dim=1)
        x = self.ln2(x + x2)
        x = self.ln3(x + self.ff(x))
        # Modified: handle variable sequence lengths
        p_len, d_len = p.shape[1], d.shape[1]
        p_out, d_out = x[:,:p_len], x[:,p_len:p_len+d_len]
        return p_out, d_out

class BilinearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wb = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wb)
        self.gp = nn.Linear(d, d); self.gd = nn.Linear(d, d)
    def forward(self, p, d):
        # Modified: pool sequences to single vectors for bilinear gate
        p_pooled = p.mean(dim=1) if p.dim() == 3 else p.squeeze(1)
        d_pooled = d.mean(dim=1) if d.dim() == 3 else d.squeeze(1)
        bil = torch.einsum("bd,dk,bk->bd", p_pooled, self.Wb, d_pooled)
        gp = torch.sigmoid(self.gp(d_pooled))
        gd = torch.sigmoid(self.gd(p_pooled))
        p_g = p_pooled * gp
        d_g = d_pooled * gd
        fuse = torch.cat([p_g, d_g, p_pooled*d_pooled, torch.abs(p_pooled-d_pooled), bil], dim=-1)
        return fuse

def batch_to_sequence(node_features, batch, max_nodes=None):
    """Convert batched node features to padded sequences for each graph."""
    batch_size = batch.max().item() + 1
    lengths = torch.bincount(batch, minlength=batch_size)  # CHANGED: added to compute sizes

    if max_nodes is None:
        # Find max number of nodes in any graph
        max_nodes = lengths.max().item()  # CHANGED: use lengths instead of redefining cap
    
    # Initialize padded tensor
    d = node_features.shape[-1]
    seq = torch.zeros(batch_size, max_nodes, d, device=node_features.device)

    # Fill in the sequences
    for b in range(batch_size):
        nodes_b = node_features[batch == b]
        n = min(nodes_b.shape[0], max_nodes)        # CHANGED: truncate if needed
        if n > 0:
            seq[b, :n] = nodes_b[:n]               # CHANGED: safe assign

    return seq

class DrugGeneEncoder(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=None, freeze_gnn=True, 
                 gnn_type='gcn', use_xattn=False, return_sequences=False, max_nodes=50, max_genes=5000, gene_pool_factor=8):
        super().__init__()
        self.use_xattn = use_xattn
        self.return_sequences = return_sequences  # Added: control sequence vs pooled output
        self.max_nodes = max_nodes  # Added: max nodes for drug graphs
        self.max_genes = max_genes  # Added: max length for gene sequences
        self.gene_pool_factor = gene_pool_factor  # Added: pooling factor for gene sequences
        
        # Modified: use 'none' pooling to get node-level features when needed
        pooling_type = 'none' if return_sequences else 'attention'
        self.gnn = GNN_graphpred(5, emb_dim, emb_dim, graph_pooling=pooling_type, gnn_type=gnn_type)
        
        if gnn_ckpt: self.gnn.from_pretrained(gnn_ckpt)
        if freeze_gnn:
            for p in self.gnn.parameters(): p.requires_grad = False
            self.gnn.eval()
        
        self.pproj = nn.Sequential(nn.Linear(dim_p, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.dproj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.gene_pool = WindowAttnPool(d_model, factor=gene_pool_factor) if gene_pool_factor and gene_pool_factor > 1 else None
        if use_xattn: 
            self.xblk = CrossAttnBlock(d_model, nheads=2, ff=2)
            if return_sequences:
                # Added: pooling after cross-attention for sequence outputs
                self.p_pool = nn.Linear(d_model, d_model)
                self.d_pool = nn.Linear(d_model, d_model)
        
        self.bgate = BilinearGate(d_model)
        self.trunk = nn.Sequential(
            nn.Linear(5*d_model, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512)
        )
    
    def forward(self, p_vec, batch_data):
        device = p_vec.device
        batch_data = batch_data.to(device)
        
        # Handle gene embeddings - can be sequences or pooled vectors
        if p_vec.dim() == 2:  # [B, dim_p] - pooled vectors
            p = self.pproj(p_vec).unsqueeze(1)  # [B, 1, d_model]
        else:  # [B, seq_len, dim_p]
            p = self.pproj(p_vec)                      # [B, Tp, d_model]
            if self.gene_pool is not None and p.size(1) > 1:
                p = self.gene_pool(p)  
        
        # Get drug embeddings
        if self.return_sequences:
            # Modified: get node-level features and convert to sequences
            node_features, batch = self.gnn(batch_data)
            d_seq = batch_to_sequence(node_features, batch, self.max_nodes)  # [B, max_nodes, emb_dim]
            d = self.dproj(d_seq)  # [B, max_nodes, d_model]
        else:
            d_vec = self.gnn(batch_data)  # [B, emb_dim]
            d = self.dproj(d_vec).unsqueeze(1)  # [B, 1, d_model]
        
        # Cross-attention if enabled
        if self.use_xattn and hasattr(self, "xblk"):
            p, d = self.xblk(p, d)
            
            if self.return_sequences:
                # Added: learnable pooling after cross-attention
                p = torch.tanh(self.p_pool(p.mean(dim=1, keepdim=True)))  # [B, 1, d_model]
                d = torch.tanh(self.d_pool(d.mean(dim=1, keepdim=True)))  # [B, 1, d_model]
        
        # Final fusion and output
        h = self.trunk(self.bgate(p, d))
        return h

class GeneDrugTargetClassifier(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=None, freeze_gnn=True, 
                 gnn_type='gcn', use_xattn=False, n_classes=2, return_sequences=False):
        super().__init__()
        # Added: return_sequences parameter
        self.encoder = DrugGeneEncoder(dim_p, emb_dim, d_model, gnn_ckpt, freeze_gnn, 
                                       gnn_type, use_xattn, return_sequences)
        self.head = nn.Linear(512, n_classes)
    
    def forward(self, p_vec, batch_data):
        h = self.encoder(p_vec, batch_data)
        return self.head(h)  # [B,2]

class ExpressionDirectionClassifier(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=None, freeze_gnn=True, 
                 gnn_type='gcn', use_xattn=False, n_classes=2, return_sequences=False):
        super().__init__()
        # Added: return_sequences parameter
        self.encoder = DrugGeneEncoder(dim_p, emb_dim, d_model, gnn_ckpt, freeze_gnn, 
                                       gnn_type, use_xattn, return_sequences)
        self.head = nn.Linear(512, n_classes)
    
    def forward(self, p_vec, batch_data):
        h = self.encoder(p_vec, batch_data)
        return self.head(h)  # [B,2]