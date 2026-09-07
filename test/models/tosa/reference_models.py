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


def make_tiny_cnn_int8() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    """Quantized (int8) variant of tiny_cnn for TOSA i8 lowering.

    Uses eager-mode static quantization (QuantStub/DeQuantStub) so
    `torch_mlir` → TOSA emits `tensor<...xi8>` / `tosa.conv2d` i8.
    Input/output stay f32 (quantized internally). For pure i8 I/O remove
    DeQuantStub and feed `torch.quint8` inputs.
    """
    class TinyCNNInt8(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
            self.relu2 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flat = nn.Flatten()
            self.fc = nn.Linear(8, 4)
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.quant(x)
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.pool(x)
            x = self.flat(x)
            x = self.fc(x)
            x = self.dequant(x)
            return x

    model = TinyCNNInt8().eval()
    # fbgemm for x86, qnnpack for ARM – try fbgemm first
    try:
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    except Exception:
        qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    model.qconfig = qconfig
    # Fuse Conv+ReLU for better int8 lowering (optional, ignore failure)
    try:
        torch.ao.quantization.fuse_modules(
            model, [["conv1", "relu1"], ["conv2", "relu2"]], inplace=True
        )
    except Exception:
        pass
    torch.ao.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        # calibrate with float example
        model(torch.randn(1, 3, 32, 32))
    torch.ao.quantization.convert(model, inplace=True)
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
    "tiny_cnn_int8": make_tiny_cnn_int8,
    "depthwise_tiny_cnn": make_depthwise_tiny_cnn,
    "residual_tiny_cnn": make_residual_tiny_cnn,
    "tiny_attention": make_tiny_attention_block,
    "mobilenet_v2": make_mobilenet_v2,
}