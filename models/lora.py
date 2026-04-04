"""
LoRA (Low-Rank Adaptation) implementation for credit scoring transformers.
Attaches trainable A and B matrices to frozen Q and V projections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a LoRA adapter: W_out = W_frozen(x) + alpha/r * B(A(x))
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.enabled = enabled

        # Frozen base weight
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA matrices — created on same device as the frozen weight
        device = self.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialise A with kaiming uniform, B with zeros (so adapter starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.enabled and self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            return base + lora_out * self.scaling
        return base

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA into base weight and return a plain Linear."""
        merged_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight = nn.Parameter(merged_weight)
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.clone())
        return linear

    def disable_lora(self):
        self.enabled = False

    def enable_lora(self):
        self.enabled = True

    def trainable_parameters(self):
        return [self.lora_A, self.lora_B]

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.lora_alpha}"


def attach_lora_to_attention(module: nn.Module, r: int = 8, lora_alpha: float = 16.0,
                              lora_dropout: float = 0.0, target_keys=("q_proj", "v_proj")):
    """
    Recursively replace target Linear layers with LoRALinear adapters.
    Returns a dict mapping layer name → LoRALinear for easy parameter access.
    """
    lora_layers = {}
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(k in name for k in target_keys):
            lora_layer = LoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            setattr(module, name, lora_layer)
            lora_layers[name] = lora_layer
        else:
            sub_lora = attach_lora_to_attention(child, r, lora_alpha, lora_dropout, target_keys)
            for k, v in sub_lora.items():
                lora_layers[f"{name}.{k}"] = v
    return lora_layers


def freeze_non_lora(model: nn.Module):
    """Freeze everything except LoRA parameters."""
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def merge_lora_into_model(model: nn.Module) -> nn.Module:
    """
    Traverse model and replace all LoRALinear layers with their merged nn.Linear equivalents.
    Operates in-place but also returns the model.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            setattr(model, name, child.merge_weights())
        else:
            merge_lora_into_model(child)
    return model
