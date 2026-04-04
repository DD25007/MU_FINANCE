"""
FT-Transformer (Feature Tokenization Transformer) — Gorishniy et al. NeurIPS 2021.
Each tabular feature is embedded into a token, then processed by standard transformer blocks.
LoRA adapters attach to Q and V projections in every attention layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from models.lora import LoRALinear, attach_lora_to_attention, freeze_non_lora


class NumericEmbedding(nn.Module):
    """Embeds each numeric feature as a learnable linear projection + bias token."""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.empty(num_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalEmbedding(nn.Module):
    """One embedding table per categorical feature."""

    def __init__(self, cat_dims: List[int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, d_model) for dim in cat_dims]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_cat_features)  — integer indices
        return torch.stack(
            [emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )


class CLSToken(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.token.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # Name projections explicitly so LoRA can find them
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        # Standard key and output projections (frozen during unlearning)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom attention using explicit Q/K/V projections so LoRA applies
        residual = x
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        # Reshape for multi-head attention manually
        B, T, D = Q.shape
        n_heads = self.attn.num_heads
        head_dim = D // n_heads

        def split_heads(t):
            return t.view(B, T, n_heads, head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        scale = math.sqrt(head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        x = residual + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer for binary credit scoring.

    Args:
        num_num_features: number of numeric/continuous features
        cat_dims: list of cardinalities for each categorical feature
        d_model: token embedding dimension
        n_heads: attention heads
        n_layers: transformer depth
        ffn_factor: FFN hidden dim = ffn_factor * d_model
        dropout: dropout rate
    """

    def __init__(
        self,
        num_num_features: int = 0,
        cat_dims: Optional[List[int]] = None,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        cat_dims = cat_dims or []

        # Feature tokenisers
        if num_num_features > 0:
            self.num_embed = NumericEmbedding(num_num_features, d_model)
        else:
            self.num_embed = None

        if cat_dims:
            self.cat_embed = CategoricalEmbedding(cat_dims, d_model)
        else:
            self.cat_embed = None

        self.cls_token = CLSToken(d_model)

        ffn_dim = d_model * ffn_factor
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # binary classification logit

    def tokenize(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = []
        if x_num is not None and self.num_embed is not None:
            tokens.append(self.num_embed(x_num))  # (B, num_num, D)
        if x_cat is not None and self.cat_embed is not None:
            tokens.append(self.cat_embed(x_cat))  # (B, num_cat, D)
        return torch.cat(tokens, dim=1)  # (B, num_features, D)

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor] = None):
        tokens = self.tokenize(x_num, x_cat)  # (B, F, D)
        tokens = self.cls_token(tokens)        # (B, F+1, D)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls_repr = tokens[:, 0]               # CLS token
        logits = self.head(cls_repr).squeeze(-1)
        return logits

    def attach_lora(self, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0):
        """
        Attach LoRA adapters to all Q and V projections in transformer blocks.
        Freezes everything else. Returns dict of LoRA layers.
        """
        lora_layers = {}
        for i, block in enumerate(self.blocks):
            # Replace q_proj
            block.q_proj = LoRALinear(block.q_proj, r=r, lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout)
            # Replace v_proj
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
