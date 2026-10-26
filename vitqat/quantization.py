"""Uniform affine fake quantization; floating point kernels, not packed inference."""

import torch
from torch import nn
from torch.nn import functional as F
from .cropr_adapter import CrossAttention
from torch.ao.quantization import (
    FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
)


class Quantizer(nn.Module):
    def __init__(self, bits, weight=False):
        super().__init__()
        if bits not in (2, 3, 4, 8):
            raise ValueError("Supported bit widths: 2, 3, 4, 8")
        self.bits = bits
        self.frozen = False
        self.calibrating = False
        kwargs = dict(quant_min=0, quant_max=2**bits - 1, dtype=torch.quint8)
        if weight:
            kwargs.update(observer=MovingAveragePerChannelMinMaxObserver,
                          qscheme=torch.per_channel_affine, ch_axis=0, averaging_constant=1.0)
        else:
            kwargs.update(observer=MovingAverageMinMaxObserver,
                          qscheme=torch.per_tensor_affine, averaging_constant=0.01)
        self.fake_quant = FakeQuantize(**kwargs)

    def forward(self, x):
        self.fake_quant.enable_observer((self.training or self.calibrating) and not self.frozen)
        observed_min = self.fake_quant.activation_post_process.min_val
        if not self.training and not self.calibrating and (observed_min.numel() == 0 or not torch.isfinite(observed_min).all()):
            raise RuntimeError("Quantized evaluation requires trained/calibrated observer state")
        return self.fake_quant(x)


class QATLinear(nn.Linear):
    def forward(self, x):
        return self.output_quant(F.linear(self.input_quant(x), self.weight_quant(self.weight), self.bias))


class QATConv2d(nn.Conv2d):
    def forward(self, x):
        return self.output_quant(self._conv_forward(self.input_quant(x), self.weight_quant(self.weight), self.bias))


class QATAttention(nn.Module):
    """Quantize both attention matmul operands, retaining the timm parameter names."""
    def __init__(self, original, activation_bits):
        super().__init__()
        for name in ("qkv", "q_norm", "k_norm", "attn_drop", "proj", "proj_drop"):
            setattr(self, name, getattr(original, name))
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.scale = original.scale
        self.q_quant = Quantizer(activation_bits)
        self.k_quant = Quantizer(activation_bits)
        self.v_quant = Quantizer(activation_bits)
        self.prob_quant = Quantizer(activation_bits)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_quant(self.q_norm(q))
        k = self.k_quant(self.k_norm(k))
        v = self.v_quant(v)
        prob = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        prob = self.prob_quant(self.attn_drop(prob))
        x = (prob @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class QATCrossAttention(CrossAttention):
    """Quantized scorer/aggregation; the original Cropr source stays unchanged."""

    def _query_key(self, x):
        b, n = x.shape[:2]
        queries = self.query_quant(self.queries).expand(b, -1, -1)
        q = self.q_quant(self.q(queries)).reshape(b, self.num_queries, -1, self.head_dim).transpose(1, 2)
        k = self.k_quant(self.k(x)).reshape(b, n, -1, self.head_dim).transpose(1, 2)
        return q, k

    def forward_scorer(self, x):
        q, k = self._query_key(self.attn_norm(x))
        return (q @ k.transpose(-2, -1)).sum((1, 2))

    def forward(self, x):
        b, n, d = x.shape
        x = self.attn_norm(x)
        q, k = self._query_key(x)
        v = self.v_quant(self.v(x)).reshape(b, n, -1, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1)
        prob = self.prob_quant((attn * self.scale).softmax(dim=-1))
        x = (prob @ v).transpose(1, 2).reshape(b, self.num_queries, d).squeeze(1)
        x = self.proj(x)
        if hasattr(self, "mlp"):
            x = x + self.mlp(self.mlp_norm(x))
        return x, attn.sum((1, 2))


def prepare_qat(model, weight_bits, activation_bits, scope="all"):

    if scope not in ("all", "backbone"):
        raise ValueError("QAT scope must be all or backbone")
    # Attach quantizers to existing parameters: no reinitialization or RNG consumption.
    for block in model.blocks:
        block.attn = QATAttention(block.attn, activation_bits)
    for name, module in list(model.named_modules()):
        if scope == "backbone" and name.startswith("cropr."):
            continue
        if isinstance(module, CrossAttention):
            module.__class__ = QATCrossAttention
            module.query_quant = Quantizer(weight_bits, weight=True)
            module.q_quant = Quantizer(activation_bits)
            module.k_quant = Quantizer(activation_bits)
            module.v_quant = Quantizer(activation_bits)
            module.prob_quant = Quantizer(activation_bits)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.__class__ = QATLinear if isinstance(module, nn.Linear) else QATConv2d
            module.input_quant = Quantizer(activation_bits)
            module.weight_quant = Quantizer(weight_bits, weight=True)
            module.output_quant = Quantizer(activation_bits)
    return model


def freeze_observers(model, frozen):
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.frozen = frozen
