# model_xattn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from gnn import GNN_graphpred

# -------------------------
# Blocks (unchanged where possible)
# -------------------------

class CrossAttnBlock(nn.Module):
    def __init__(self, d, nheads=4, ff=4):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff*d), nn.GELU(), nn.Linear(ff*d, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)

    def forward(self, p, d, p_mask=None, d_mask=None):
        x = torch.cat([p, d], dim=1)  # [B, Tp+Td, D]
        if p_mask is None:
            p_mask = torch.zeros(p.size(0), p.size(1), dtype=torch.bool, device=p.device)
        if d_mask is None:
            d_mask = torch.zeros(d.size(0), d.size(1), dtype=torch.bool, device=d.device)
        concat_mask = torch.cat([p_mask, d_mask], dim=1)  # True=PAD

        x = self.ln1(x + self.sa(x, x, x, key_padding_mask=concat_mask, need_weights=False)[0])
        p2 = self.ca(p, d, d, key_padding_mask=d_mask, need_weights=False)[0]
        d2 = self.ca(d, p, p, key_padding_mask=p_mask, need_weights=False)[0]
        x2 = torch.cat([p2, d2], dim=1)
        x = self.ln2(x + x2)
        x = self.ln3(x + self.ff(x))

        Tp = p.size(1)
        p_out, d_out = x[:, :Tp], x[:, Tp:]
        return p_out, d_out


class BilinearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wb = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wb)
        self.gp = nn.Linear(d, d); self.gd = nn.Linear(d, d)
    def forward(self, p, d):
        # Pool sequences to single vectors for bilinear gate
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
    B = int(batch.max().item()) + 1
    lengths = torch.bincount(batch, minlength=B)
    if max_nodes is None:
        max_nodes = int(lengths.max().item())

    d = node_features.size(-1)
    seq = node_features.new_zeros(B, max_nodes, d)
    pad_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=node_features.device)

    for b in range(B):
        idx = (batch == b).nonzero(as_tuple=False).squeeze(1)
        n = min(int(lengths[b].item()), max_nodes)
        if n > 0:
            seq[b, :n] = node_features.index_select(0, idx)[:n]
            pad_mask[b, :n] = False

    return seq, pad_mask  # True=PAD


class MLPHead(nn.Module):
    def __init__(self, in_dim=512, hidden=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Encoder (WindowAttnPool removed)
# -------------------------

class DrugGeneEncoder(nn.Module):
    def __init__(
        self,
        dim_p=768,
        emb_dim=300,
        d_model=512,
        gnn_ckpt=None,
        freeze_gnn=True,
        gnn_type='gcn',
        use_xattn=False,
        return_sequences=False,
        max_nodes=50,
        max_genes=5000,
        gene_pool_factor=8  # kept only for API-compat; unused now
    ):
        super().__init__()
        self.use_xattn = use_xattn
        self.return_sequences = return_sequences
        self.max_nodes = max_nodes
        self.max_genes = max_genes
        self.freeze_gnn = freeze_gnn

        # Use 'none' pooling if sequence path, else attention (same as before)
        pooling_type = 'none' if return_sequences else 'attention'
        self.gnn = GNN_graphpred(5, emb_dim, emb_dim, graph_pooling=pooling_type, gnn_type=gnn_type)
        if gnn_ckpt: 
            self.gnn.from_pretrained(gnn_ckpt)
        if freeze_gnn:
            for p in self.gnn.parameters(): 
                p.requires_grad = False
            self.gnn.eval()

        # Light input normalization to align scales
        self.p_in = nn.LayerNorm(dim_p)
        self.d_in = nn.LayerNorm(emb_dim)

        self.pproj = nn.Sequential(nn.Linear(dim_p, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.dproj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))

        if use_xattn:
            self.xblk = CrossAttnBlock(d_model, nheads=2, ff=2)
            if return_sequences:
                self.p_pool = nn.Linear(d_model, d_model)
                self.d_pool = nn.Linear(d_model, d_model)

        self.bgate = BilinearGate(d_model)
        self.trunk = nn.Sequential(
            nn.Linear(5*d_model, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512)
        )

    def forward(self, p_vec, batch_data=None, drug_node_seq=None, p_mask=None, d_mask=None):
        device = p_vec.device

        # ---- gene path (no WindowAttnPool; keep shapes/masks stable) ----
        if p_vec.dim() == 2:
            # pooled gene vector -> [B,1,D]
            p = self.pproj(self.p_in(p_vec)).unsqueeze(1)
            if p_mask is None:
                p_mask = torch.zeros(p.size(0), 1, dtype=torch.bool, device=device)
        else:
            # sequence of gene embeddings -> [B,Tp,D]
            p = self.pproj(self.p_in(p_vec))
            if p_mask is None:
                p_mask = torch.zeros(p.size(0), p.size(1), dtype=torch.bool, device=device)

        # ---- drug path ----
        if self.return_sequences:
            assert batch_data is not None, "Provide batch_data (PyG Batch/Data) when return_sequences=True"
            batch_data = batch_data.to(device)
            if self.freeze_gnn:
                with torch.no_grad():
                    node_features, pyg_batch = self.gnn(batch_data)  # [N,D_emb], batch idx
            else:
                node_features, pyg_batch = self.gnn(batch_data)
            d_seq, d_mask = batch_to_sequence(node_features, pyg_batch, self.max_nodes)  # [B,Td,emb_dim], [B,Td]
            d = self.dproj(self.d_in(d_seq))  # [B,Td,D]
        else:
            if batch_data is not None:
                batch_data = batch_data.to(device)
                if self.freeze_gnn:
                    with torch.no_grad():
                        d_vec = self.gnn(batch_data)   # [B,emb_dim]
                else:
                    d_vec = self.gnn(batch_data)
            else:
                # fallback: use provided node sequence if any
                d_vec = drug_node_seq.mean(dim=1) if drug_node_seq is not None else None
                assert d_vec is not None, "Provide batch_data or a pooled drug vector"
            d = self.dproj(self.d_in(d_vec)).unsqueeze(1)           # [B,1,D]
            d_mask = torch.zeros(d.size(0), 1, dtype=torch.bool, device=device)

        # ---- optional cross-attention ----
        if self.use_xattn and hasattr(self, "xblk"):
            p, d = self.xblk(p, d, p_mask=p_mask, d_mask=d_mask)
            if self.return_sequences:
                # keep a simple learned pooling after xattn
                p = torch.tanh(self.p_pool(p.mean(dim=1, keepdim=True)))
                d = torch.tanh(self.d_pool(d.mean(dim=1, keepdim=True)))

        # ---- fuse & project ----
        h = self.trunk(self.bgate(p, d))  # [B,512]
        return h

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_gnn:
            self.gnn.eval()
        return self


class GeneDrugTargetClassifier(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=None, freeze_gnn=True, 
                 gnn_type='gcn', use_xattn=False, n_classes=2, return_sequences=False):
        super().__init__()
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
        self.encoder = DrugGeneEncoder(dim_p, emb_dim, d_model, gnn_ckpt, freeze_gnn,
                                       gnn_type, use_xattn, return_sequences)
        self.head = MLPHead(in_dim=512, hidden=256, n_classes=n_classes, dropout=0.3)
    def forward(self, p_vec, batch_data=None, drug_node_seq=None, p_mask=None, d_mask=None):
        h = self.encoder(
            p_vec,
            batch_data=batch_data,
            drug_node_seq=None,
            p_mask=p_mask,
            d_mask=d_mask
        )
        return self.head(h)  # (no trailing comma)
