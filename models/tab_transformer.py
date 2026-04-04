"""
TabTransformer — Huang et al. 2020.
Applies transformer attention only to categorical features.
Numerical features are concatenated directly to the CLS output.
Used as a secondary/ablation model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from models.lora import LoRALinear, freeze_non_lora


class TabTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        n_heads = self.n_heads
        head_dim = D // n_heads
        residual = x
        x_norm = self.norm1(x)

        Q = self.q_proj(x_norm).view(B, T, n_heads, head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, T, n_heads, head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, T, n_heads, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim)
        attn = F.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        x = residual + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class TabTransformer(nn.Module):
    """
    TabTransformer for binary credit scoring.

    Categorical features → embedding → transformer → flatten
    Numeric features → BatchNorm → concat with cat output → MLP head
    """

    def __init__(
        self,
        num_num_features: int = 0,
        cat_dims: Optional[List[int]] = None,
        d_model: int = 32,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_factor: int = 4,
        dropout: float = 0.1,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        cat_dims = cat_dims or []
        self.num_num_features = num_num_features
        self.num_cat_features = len(cat_dims)
        self.d_model = d_model

        if cat_dims:
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(dim + 1, d_model) for dim in cat_dims]
            )
        else:
            self.cat_embeddings = nn.ModuleList()

        ffn_dim = d_model * ffn_factor
        self.blocks = nn.ModuleList(
            [TabTransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        if num_num_features > 0:
            self.num_bn = nn.BatchNorm1d(num_num_features)
        else:
            self.num_bn = None

        cat_output_dim = self.num_cat_features * d_model
        total_dim = cat_output_dim + num_num_features
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor] = None):
        parts = []

        if x_cat is not None and self.num_cat_features > 0:
            cat_tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
            )  # (B, num_cat, d_model)
            for block in self.blocks:
                cat_tokens = block(cat_tokens)
            cat_tokens = self.norm(cat_tokens)
            cat_flat = cat_tokens.view(cat_tokens.size(0), -1)  # (B, num_cat * d_model)
            parts.append(cat_flat)

        if x_num is not None and self.num_num_features > 0:
            x_num_normed = self.num_bn(x_num) if self.num_bn else x_num
            parts.append(x_num_normed)

        combined = torch.cat(parts, dim=1)
        return self.mlp(combined).squeeze(-1)

    def attach_lora(self, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0):
        lora_layers = {}
        for i, block in enumerate(self.blocks):
            block.q_proj = LoRALinear(block.q_proj, r=r, lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout)
            block.v_proj = LoRALinear(block.v_proj, r=r, lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout)
            lora_layers[f"block.{i}.q_proj"] = block.q_proj
            lora_layers[f"block.{i}.v_proj"] = block.v_proj
        freeze_non_lora(self)
        return lora_layers

    def get_lora_params(self):
        params = []
        for block in self.blocks:
            for proj in [block.q_proj, block.v_proj]:
                if isinstance(proj, LoRALinear):
                    params.extend(proj.trainable_parameters())
        return params
