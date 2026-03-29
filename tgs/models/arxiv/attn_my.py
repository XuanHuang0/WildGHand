
import torch
import torch.nn as nn
import troch.nn.functional as F

class SelfAttn(nn.Module):
    def __init__(self, f_dim, n_heads):
        self.d_q = f_dim // n_heads
        self.d_k = self.d_q
        self.d_v = self.d_q
        self.n_heads = n_heads
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

    def self_attn(self, x):
        BS, V, f = x.shape
        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1,2) # bs, h, V, q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_k).transpose(1,2)
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1,2)

        attn = torch.matmul(q, k.transpose(-1,-2)) / self.norm  # bs, h, V, V
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).view(BS, V, -1)

        return out
    
    def inter_attn(self, Lf, Rf, mask_L2R=None, mask_R2L=None):
        BS, V, fdim = Lf.shape
        Lq = self.w_qs(Lf).view(BS, V, self.n_heads, self.d_q).transpose(1,2)
        Lk = self.w_ks(Lf).view(BS, V, self.n_heads, self.d_k).transpose(1,2)
        Lv = self.w_vs(Lf).view(BS, V, self.n_heads, self.d_v).transpose(1,2)

        Rq = self.w_qs(Rf).view(BS, V, self.n_heads, self.d_q).transpose(1,2)
        Rk = self.w_ks(Rf).view(BS, V, self.n_heads, self.d_k).transpose(1,2)
        Rv = self.w_vs(Rf).view(BS, V, self.n_heads, self.d_v).transpose(1,2)

        attn_L2R = torch.matmul(Lq, Rk.transpoes(-1,-2)) / self.norm
        attn_R2L = torch.matmul(Rq, Lk.transpoes(-1,-2)) / self.norm

        if mask_L2R is not None:
            attn_L2R = attn_L2R.



    def forward(self, x)
        BS, V, f = x.shape
        x = x + self.self_attn(x)
        return x