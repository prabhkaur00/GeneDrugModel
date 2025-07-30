import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime

LOG_FILE = "/mnt/data/logs.txt"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")


class CrossAttnBlock(nn.Module):
    def __init__(self, d, nheads=4, ff=4):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff*d), nn.GELU(), nn.Linear(ff*d, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)

    def forward(self, p, d):                    # p,d: (B,1,D)
        x = torch.cat([p,d], dim=1)             # (B,2,D)
        x = self.ln1(x + self.sa(x,x,x,need_weights=False)[0])
        p2 = self.ca(p, d, d, need_weights=False)[0]
        d2 = self.ca(d, p, p, need_weights=False)[0]
        x2 = torch.cat([p2,d2], dim=1)
        x = self.ln2(x + x2)
        x = self.ln3(x + self.ff(x))
        p_out, d_out = x[:,0:1], x[:,1:2]
        log(f"[XATTN] p_out: {p_out.shape}, d_out: {d_out.shape}")
        return p_out, d_out

class BilinearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wb = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wb)
        self.gp = nn.Linear(d, d); self.gd = nn.Linear(d, d)

    def forward(self, p, d):  # (B,1,D)
        p, d = p.squeeze(1), d.squeeze(1)
        bil = torch.einsum("bd,dk,bk->bd", p, self.Wb, d)
        gp = torch.sigmoid(self.gp(d))
        gd = torch.sigmoid(self.gd(p))
        p_g = p * gp
        d_g = d * gd
        fuse = torch.cat([p_g, d_g, p*d, torch.abs(p-d), bil], dim=-1)
        log(f"[BILINEAR] fuse shape: {fuse.shape}")
        return fuse

class TwoHeadConditional(nn.Module):
    def __init__(self, dim_p=768, dim_d=300, d_model=512, n_targets=10, n_dirs=3):
        super().__init__()
        self.pproj = nn.Linear(dim_p, d_model)
        self.dproj = nn.Linear(dim_d, d_model)
        self.xblk  = CrossAttnBlock(d_model, nheads=4, ff=4)
        self.bgate = BilinearGate(d_model)
        fuse_dim   = d_model*4 + d_model
        self.trunk = nn.Sequential(nn.Linear(fuse_dim, 1024), nn.GELU(),
                                   nn.Dropout(0.2), nn.Linear(1024, 512), nn.GELU())
        self.head_t = nn.Linear(512, n_targets)
        self.target_emb = nn.Embedding(n_targets, 64)
        self.head_d = nn.Linear(512+64, n_dirs)

    def forward(self, p_vec, d_vec):
        log(f"[FWD] input p_vec: {p_vec.shape}, d_vec: {d_vec.shape}")
        p = self.pproj(p_vec).unsqueeze(1)    # (B,1,512)
        d = self.dproj(d_vec).unsqueeze(1)    # (B,1,512)
        log(f"[PROJ] p_proj: {p.shape}, d_proj: {d.shape}")

        p, d = self.xblk(p, d)                # Cross-attn block
        fuse = self.bgate(p, d)               # Bilinear + gates

        h = self.trunk(fuse)                  # (B, 512)
        logit_t = self.head_t(h)              # (B, n_targets)
        soft_t  = F.softmax(logit_t, dim=-1)
        t_ctx   = torch.matmul(soft_t, self.target_emb.weight)  # (B, 64)

        log(f"[HEAD_T] logit_t: {logit_t.shape}, soft_t: {soft_t.shape}")
        logit_d = self.head_d(torch.cat([h, t_ctx], dim=-1))    # (B, n_dirs)
        log(f"[HEAD_D] logit_d: {logit_d.shape}")

        return logit_t, logit_d