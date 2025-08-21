import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import datetime
from gnn import GNN_graphpred

LOG_FILE = "/mnt/data/cls/logs.txt"

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
    
class StageBJoint(nn.Module):
    """
    Joint single-label Stage-B model:
      - Target head (2-way softmax): [expression, methylation]
      - Direction head (3-way softmax): [increase, decrease, none],
        conditioned on the target distribution via a small embedding.

    Inputs
      p_vec:     [B, dim_p]  pooled DNABERT2 gene vector
      batch_data: torch_geometric Batch for the drug graph

    Outputs
      logit_t: [B, 2]
      logit_d: [B, 3]
    """
    def __init__(self,
                 dim_p=768,
                 emb_dim=300,
                 d_model=512,
                 n_dirs=3,
                 gnn_ckpt=None,
                 freeze_gnn=True,
                 gnn_type='gcn',
                 use_xattn=False):
        super().__init__()
        self.use_xattn = use_xattn
        self.gnn_type = gnn_type
        # ---- drug encoder (frozen) ----
        self.gnn = self.create_gnn(gnn_ckpt, freeze_gnn, emb_dim)

        # ---- projections to shared space ----
        self.pproj = nn.Sequential(
            nn.Linear(dim_p, d_model), nn.ReLU(), nn.LayerNorm(d_model)
        )
        self.dproj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model)
        )

        if use_xattn:
            self.xblk = CrossAttnBlock(d_model, nheads=2, ff=2)

        # ---- fusion + trunk ----
        self.bgate = BilinearGate(d_model)
        fuse_dim = 5 * d_model  # [p_g, d_g, p*d, |p-d|, bil]
        self.trunk = nn.Sequential(
            nn.Linear(fuse_dim, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512)
        )

        # ---- heads ----
        self.head_t = nn.Linear(512, 2)        # (expression, methylation)
        self.target_emb = nn.Embedding(2, 64)  # conditioning for direction
        self.head_d = nn.Linear(512 + 64, n_dirs)  # (increase, decrease, none)

    def create_gnn(self, model_path, freeze, emb_dim):
        # mirror your original create_gnn signature/behavior
        gnn = GNN_graphpred(5, emb_dim, emb_dim, graph_pooling='attention', gnn_type=self.gnn_type)
        if model_path:
            gnn.from_pretrained(model_path)
        if freeze:
            for p in gnn.parameters():
                p.requires_grad = False
            gnn.eval()
        return gnn

    def forward(self, p_vec, batch_data):
        device = p_vec.device
        batch_data = batch_data.to(device)

        # drug vector from (frozen) GNN
        d_vec = self.gnn(batch_data)                         # [B, emb_dim]

        # project both sides, add singleton seq dim
        p = self.pproj(p_vec).unsqueeze(1)                   # [B,1,d_model]
        d = self.dproj(d_vec).unsqueeze(1)                   # [B,1,d_model]

        # optional tiny mixer over two tokens (often leave OFF for pooled vectors)
        if self.use_xattn:
            p, d = self.xblk(p, d)                           # [B,1,d_model] each

        # fusion + trunk
        fuse = self.bgate(p, d)                              # [B, 5*d_model]
        h = self.trunk(fuse)                                 # [B, 512]

        # target (2-way softmax)
        logit_t = self.head_t(h)                             # [B,2]
        t_prob = F.softmax(logit_t, dim=-1)                  # [B,2]

        # direction conditioned on target distribution
        t_ctx = t_prob @ self.target_emb.weight              # [B,64]
        logit_d = self.head_d(torch.cat([h, t_ctx], dim=-1)) # [B,3]

        return logit_t, logit_d