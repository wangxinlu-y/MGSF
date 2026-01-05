import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttention(nn.Module):
    def __init__(self, dim, num_tokens, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        hidden_dim=512
        self.hyper_token = nn.Sequential(
            nn.Linear(num_tokens * self.head_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.hyper_dim = nn.Sequential(
            nn.Linear(num_tokens * self.head_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.last_token_ratio = 0.0
        self.last_dim_ratio = 0.0
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

        self.token_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, 1)
        )

        self.dim_mlp = nn.Sequential(
            nn.Linear(num_tokens, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, 1)
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        seq_feat = q[:, 0, :, :].reshape(B, -1)

        token_ratio = self.hyper_token(seq_feat).unsqueeze(1)
        dim_ratio = self.hyper_dim(seq_feat).unsqueeze(1)
        self.last_token_ratio = token_ratio.detach().mean().item()
        self.last_dim_ratio = dim_ratio.detach().mean().item()

        token_scores = self.token_mlp(q).squeeze(-1)

        k_token = int((token_scores.shape[-1] * token_ratio).clamp(
            min=1, max=token_scores.shape[-1]).max().item())

        index = torch.topk(token_scores, k=k_token, dim=-1)[1]
        mask = torch.zeros_like(token_scores)
        mask.scatter_(-1, index, 1.)
        token_scores = torch.where(mask > 0, token_scores, torch.full_like(token_scores, float('-inf')))
        token_weights = F.softmax(token_scores, dim=-1).unsqueeze(-1)

        q_transposed = q.permute(0, 1, 3, 2)
        dim_scores = self.dim_mlp(q_transposed).squeeze(-1)

        k_dim = int((dim_scores.shape[-1] * dim_ratio).clamp(
            min=1, max=dim_scores.shape[-1]).max().item())

        index = torch.topk(dim_scores, k=k_dim, dim=-1)[1]
        mask = torch.zeros_like(dim_scores)
        mask.scatter_(-1, index, 1.)
        dim_scores = torch.where(mask > 0, dim_scores, torch.full_like(dim_scores, float('-inf')))
        dim_weights = F.softmax(dim_scores, dim=-1).unsqueeze(2)

        q_token = q * token_weights
        q_dim = q * dim_weights
        q_fused = self.fusion_alpha * q_token + (1 - self.fusion_alpha) * q_dim

        attn = (q_fused @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class ViTWithDualAttention(nn.Module):
    def __init__(self, vit_model, num_heads=12):
        super().__init__()
        self.vit = vit_model
        embed_dim = vit_model.embed_dim

        num_tokens = vit_model.patch_embed.num_patches + 1

        self.dual_modules = nn.ModuleList([
            DualAttention(dim=embed_dim, num_heads=num_heads, num_tokens=num_tokens)
            for _ in range(len(vit_model.blocks) - 1)
        ])
        self.dual_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(len(vit_model.blocks) - 1)
        ])
        self._freeze_vit()

    def _freeze_vit(self):
        for name,param in self.vit.named_parameters():
            if "head" not in name:
               param.requires_grad = False

    def forward_features(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i < len(self.dual_modules):
               x = x + self.dual_modules[i](self.dual_norms[i](x))
        x = self.vit.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.vit.head(x)
        return x


