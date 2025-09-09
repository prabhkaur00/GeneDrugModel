import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from gnn import GNN_graphpred

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
        p_out, d_out = x[:,0:1], x[:,1:2]
        return p_out, d_out

class BilinearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wb = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wb)
        self.gp = nn.Linear(d, d); self.gd = nn.Linear(d, d)
    def forward(self, p, d):
        p, d = p.squeeze(1), d.squeeze(1)
        bil = torch.einsum("bd,dk,bk->bd", p, self.Wb, d)
        gp = torch.sigmoid(self.gp(d))
        gd = torch.sigmoid(self.gd(p))
        p_g = p * gp
        d_g = d * gd
        fuse = torch.cat([p_g, d_g, p*d, torch.abs(p-d), bil], dim=-1)
        return fuse

class DrugGeneEncoder(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=None, freeze_gnn=True, gnn_type='gcn', use_xattn=False):
        super().__init__()
        self.use_xattn = use_xattn
        self.gnn = GNN_graphpred(5, emb_dim, emb_dim, graph_pooling='attention', gnn_type=gnn_type)
        if gnn_ckpt: self.gnn.from_pretrained(gnn_ckpt)
        if freeze_gnn:
            for p in self.gnn.parameters(): p.requires_grad = False
            self.gnn.eval()
        self.pproj = nn.Sequential(nn.Linear(dim_p, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.dproj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        if use_xattn: self.xblk = CrossAttnBlock(d_model, nheads=2, ff=2)
        self.bgate = BilinearGate(d_model)
        self.trunk = nn.Sequential(
            nn.Linear(5*d_model, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512)
        )
    def forward(self, p_vec, batch_data):
        device = p_vec.device
        batch_data = batch_data.to(device)
        d_vec = self.gnn(batch_data)
        p = self.pproj(p_vec).unsqueeze(1)
        d = self.dproj(d_vec).unsqueeze(1)
        if getattr(self, "xblk", None) is not None: p, d = self.xblk(p, d)
        h = self.trunk(self.bgate(p, d))
        return h

class GeneDrugTargetClassifier(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=None, freeze_gnn=True, gnn_type='gcn', use_xattn=False, n_classes=2):
        super().__init__()
        self.encoder = DrugGeneEncoder(dim_p, emb_dim, d_model, gnn_ckpt, freeze_gnn, gnn_type, use_xattn)
        self.head = nn.Linear(512, n_classes)
    def forward(self, p_vec, batch_data):
        h = self.encoder(p_vec, batch_data)
        return self.head(h)  # [B,2]

class ExpressionDirectionClassifier(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, gnn_ckpt=None, freeze_gnn=True, gnn_type='gcn', use_xattn=False, n_classes=2):
        super().__init__()
        self.encoder = DrugGeneEncoder(dim_p, emb_dim, d_model, gnn_ckpt, freeze_gnn, gnn_type, use_xattn)
        self.head = nn.Linear(512, n_classes)
    def forward(self, p_vec, batch_data):
        h = self.encoder(p_vec, batch_data)
        return self.head(h)  # [B,2]
