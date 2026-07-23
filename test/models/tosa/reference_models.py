from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as models


def make_mlp() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).eval()
    example_inputs = (torch.randn(1, 32),)
    return model, example_inputs


def make_tiny_cnn() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 4),
    ).eval()
    example_inputs = (torch.randn(1, 3, 32, 32),)
    return model, example_inputs


def make_depthwise_tiny_cnn() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    model = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
        nn.ReLU(),
        nn.Conv2d(3, 8, kernel_size=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 4),
    ).eval()
    example_inputs = (torch.randn(1, 3, 32, 32),)
    return model, example_inputs


def make_residual_tiny_cnn() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return self.relu(x + residual)

    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        ResidualBlock(8),
        nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 4),
    ).eval()
    example_inputs = (torch.randn(1, 3, 32, 32),)
    return model, example_inputs


def make_tiny_attention_block() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    class TinyAttentionBlock(nn.Module):
        def __init__(self, embed_dim: int = 16, num_heads: int = 4):
            super().__init__()
            self.input_proj = nn.Linear(8, embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.output_proj = nn.Linear(embed_dim, 6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x)
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = x + attn_out
            x = x + self.ffn(x)
            x = x.mean(dim=1)
            return self.output_proj(x)

    model = TinyAttentionBlock().eval()
    example_inputs = (torch.randn(1, 8, 8),)
    return model, example_inputs


def make_mobilenet_v2() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    model = models.mobilenet_v2(weights=None).eval()
    example_inputs = (torch.randn(1, 3, 224, 224),)
    return model, example_inputs


MODEL_BUILDERS: dict[str, Callable[[], tuple[nn.Module, tuple[torch.Tensor, ...]]]] = {
    "mlp": make_mlp,
    "tiny_cnn": make_tiny_cnn,
    "depthwise_tiny_cnn": make_depthwise_tiny_cnn,
    "residual_tiny_cnn": make_residual_tiny_cnn,
    "tiny_attention": make_tiny_attention_block,
    "mobilenet_v2": make_mobilenet_v2,
}