import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

class CrossAttnBlock(nn.Module):
    def __init__(self, d, nheads=2, ff=2):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ca_pd = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ca_dp = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff*d), nn.GELU(), nn.Linear(ff*d, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2p = nn.LayerNorm(d)
        self.ln2d = nn.LayerNorm(d)
        self.ln3p = nn.LayerNorm(d)
        self.ln3d = nn.LayerNorm(d)

    def forward(self, p, d, p_pad=None, d_pad=None):
        x = torch.cat([p, d], dim=1)
        x = self.ln1(x + self.sa(x, x, x, need_weights=False)[0])
        Tp = p.size(1)
        p_s, d_s = x[:, :Tp], x[:, Tp:]
        p2 = self.ca_pd(p_s, d_s, d_s, key_padding_mask=d_pad, need_weights=False)[0]
        d2 = self.ca_dp(d_s, p_s, p_s, key_padding_mask=p_pad, need_weights=False)[0]
        p = self.ln2p(p_s + p2)
        d = self.ln2d(d_s + d2)
        p = self.ln3p(p + self.ff(p))
        d = self.ln3d(d + self.ff(d))
        return p, d

class BilinearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wb = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.Wb)
        self.gp = nn.Linear(d, d)
        self.gd = nn.Linear(d, d)
    def forward(self, p1, d1):
        p, d = p1.squeeze(1), d1.squeeze(1)
        bil = torch.einsum("bd,dk,bk->bd", p, self.Wb, d)
        gp = torch.sigmoid(self.gp(d))
        gd = torch.sigmoid(self.gd(p))
        p_g = p * gp
        d_g = d * gd
        return torch.cat([p_g, d_g, p*d, torch.abs(p-d), bil], dim=-1)

class DrugGeneEncoder(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512,
                 gnn=None, gnn_ckpt=None, freeze_gnn=True, use_xattn=True):
        super().__init__()
        assert gnn is not None, "Pass a GNN_graphpred instance via `gnn=`"
        self.gnn = gnn
        if gnn_ckpt:
            self.gnn.from_pretrained(gnn_ckpt)
        if freeze_gnn:
            for p in self.gnn.parameters():
                p.requires_grad = False
            self.gnn.eval()

        self.pproj = nn.Sequential(nn.Linear(dim_p, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.dproj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model))

        self.use_xattn = use_xattn
        if use_xattn:
            self.xblk = CrossAttnBlock(d_model, nheads=2, ff=2)

        self.pool_q_p = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_q_d = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

        self.bgate = BilinearGate(d_model)
        self.trunk = nn.Sequential(
            nn.Linear(5*d_model, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512)
        )

    def _node_embeddings(self, batch):
        if hasattr(self.gnn, "get_node_embeddings"):
            node_h = self.gnn.get_node_embeddings(batch)
        elif hasattr(self.gnn, "forward_node_embeddings"):
            node_h = self.gnn.forward_node_embeddings(batch)
        elif hasattr(self.gnn, "forward") and getattr(self.gnn, "graph_pooling", None) is None:
            node_h = self.gnn(batch)
        else:
            raise RuntimeError("GNN must expose per-node embeddings. Provide a method or set graph_pooling=None.")
        d_seq, valid = to_dense_batch(node_h, batch.batch)
        d_pad = ~valid
        return d_seq, d_pad

    def _infer_pad_from_zero(self, p_seq):
        return (p_seq.abs().sum(dim=-1) == 0)

    def _attn_pool(self, x, pad, query_param):
        q = query_param.expand(x.size(0), 1, x.size(-1))
        out = self.pool_attn(q, x, x, key_padding_mask=pad, need_weights=False)[0]
        return out

    def forward(self, p_seq, batch, p_pad=None):
        d_seq, d_pad = self._node_embeddings(batch)
        if p_pad is None:
            p_pad = self._infer_pad_from_zero(p_seq)

        p = self.pproj(p_seq)
        d = self.dproj(d_seq)

        if self.use_xattn:
            p, d = self.xblk(p, d, p_pad=p_pad, d_pad=d_pad)

        p1 = self._attn_pool(p, p_pad, self.pool_q_p)
        d1 = self._attn_pool(d, d_pad, self.pool_q_d)

        h = self.trunk(self.bgate(p1, d1))
        return h

class ExpressionDirectionClassifier(nn.Module):
    def __init__(self, dim_p=768, emb_dim=300, d_model=512,
                 gnn=None, gnn_ckpt=None, freeze_gnn=True, use_xattn=True, n_classes=2):
        super().__init__()
        self.encoder = DrugGeneEncoder(dim_p, emb_dim, d_model, gnn, gnn_ckpt, freeze_gnn, use_xattn)
        self.head = nn.Linear(512, n_classes)
    def forward(self, p_seq, batch, p_pad=None):
        h = self.encoder(p_seq, batch, p_pad=p_pad)
        return self.head(h)
