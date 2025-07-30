import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import datetime
from gnn import GNN_graphpred  # your GNN class

LOG_FILE = "/content/logs.txt"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

def cache_to_pyg_data(graph_dict):
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
    edge_feat = torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
    node_feat = torch.tensor(graph_dict['node_feat'], dtype=torch.long)
    num_nodes = graph_dict['num_nodes']

    data = torch_geometric.data.Data(
        x=node_feat,
        edge_index=edge_index,
        edge_attr=edge_feat,
        num_nodes=num_nodes
    )
    return data

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

class TwoHeadConditional(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512, n_targets=10, n_dirs=3,
                 gnn_ckpt=None, freeze_gnn=True):
        super().__init__()
        self.gnn = self.create_gnn(gnn_ckpt, freeze_gnn)
        self.pproj = nn.Linear(dim_p, d_model)
        self.dproj = nn.Linear(emb_dim, d_model)
        self.xblk  = CrossAttnBlock(d_model, nheads=4, ff=4)
        self.bgate = BilinearGate(d_model)
        fuse_dim   = d_model*4 + d_model
        self.trunk = nn.Sequential(nn.Linear(fuse_dim, 1024), nn.GELU(),
                                   nn.Dropout(0.2), nn.Linear(1024, 512), nn.GELU())
        self.head_t = nn.Linear(512, n_targets)
        self.target_emb = nn.Embedding(n_targets, 64)
        self.head_d = nn.Linear(512+64, n_dirs)

    def create_gnn(self, model_path, freeze):
        emb_dim = 300
        gnn = GNN_graphpred(5, emb_dim, emb_dim, graph_pooling='attention', gnn_type='gcn')
        if model_path:
            gnn.from_pretrained(model_path)
        if freeze:
            for param in gnn.parameters():
                param.requires_grad = False
            gnn.eval()
        return gnn

    def forward(self, p_vec, batch_data):
        batch_data = batch_data.to(p_vec.device)
        d_vec = self.gnn(batch_data)

        p = self.pproj(p_vec).unsqueeze(1)
        d = self.dproj(d_vec).unsqueeze(1)

        p, d = self.xblk(p, d)
        fuse = self.bgate(p, d)

        h = self.trunk(fuse)
        logit_t = self.head_t(h)
        soft_t = F.softmax(logit_t, dim=-1)
        t_ctx = torch.matmul(soft_t, self.target_emb.weight)
        logit_d = self.head_d(torch.cat([h, t_ctx], dim=-1))

        return logit_t, logit_d