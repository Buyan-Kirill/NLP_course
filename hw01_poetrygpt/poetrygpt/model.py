"""Classic and RoPE/RMSNorm/SwiGLU decoder-only GPT, randomly initialized."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    n_layers: int = 8
    n_heads: int = 6
    d_model: int = 384
    dropout: float = 0.1
    gradient_checkpointing: bool = False
    architecture: str = 'classic'
    ffn_hidden_size: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.n_layers, self.n_heads, self.d_model) <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.context_length < 2 or self.d_model % self.n_heads or not 0 <= self.dropout < 1:
            raise ValueError('Invalid context, head dimension or dropout')
        if self.architecture not in ('classic', 'modern'):
            raise ValueError('Architecture must be classic or modern')
        if self.architecture == 'modern':
            if self.d_model // self.n_heads % 2 or self.ffn_hidden_size < 1:
                raise ValueError('RoPE requires even head width and positive FFN width')
            if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
                raise ValueError('RoPE theta must be finite and greater than one')
            if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
                raise ValueError('RMSNorm epsilon must be finite and positive')


class RotaryEmbedding(nn.Module):
    """Interleaved-pair RoPE on queries and keys; positions reset per window."""
    def __init__(self, head_width, context_length, theta=10000.0):
        super().__init__()
        frequencies = theta ** (-torch.arange(0, head_width, 2, dtype=torch.float32) / head_width)
        angles = torch.outer(torch.arange(context_length, dtype=torch.float32), frequencies)
        # Derived caches are not checkpoint state; reconstruct from ModelConfig.
        self.register_buffer('cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('sin', angles.sin()[None, None], persistent=False)

    def forward(self, x):
        length = x.shape[-2]
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        cos, sin = self.cos[..., :length, :], self.sin[..., :length, :]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.dropout = cfg.n_heads, cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.projection = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.residual_dropout = nn.Dropout(cfg.dropout)
        self.rotary = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.context_length, cfg.rope_theta) if cfg.architecture == 'modern' else None

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2) for t in (q, k, v)]
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        # Input batches use right padding only. Causality prevents valid queries
        # from seeing any padding on their right; padded queries have no loss.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.residual_dropout(self.projection(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attention = Attention(cfg)
        self.expand = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.contract = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.dropout(self.contract(F.gelu(self.expand(self.norm2(x)), approximate='tanh')))


class ModernBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attention = Attention(cfg)
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden_size, bias=False)
        self.expand = nn.Linear(cfg.d_model, cfg.ffn_hidden_size, bias=False)
        self.contract = nn.Linear(cfg.ffn_hidden_size, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        normalized = self.norm2(x)
        return x + self.dropout(self.contract(F.silu(self.gate(normalized)) * self.expand(normalized)))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.positions = nn.Embedding(cfg.context_length, cfg.d_model) if cfg.architecture == 'classic' else None
        self.dropout = nn.Dropout(cfg.dropout)
        block_type = Block if cfg.architecture == 'classic' else ModernBlock
        self.blocks = nn.ModuleList([block_type(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model) if cfg.architecture == 'classic' else nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.apply(self._initialize)
        # Residual paths start smaller as depth increases. Output weights are
        # tied via F.linear, so the token matrix is initialized exactly once.
        for block in self.blocks:
            for weight in (block.attention.projection.weight, block.contract.weight):
                nn.init.normal_(weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context_length:
            raise ValueError('Expected [batch, time] within configured context')
        x = self.tokens(ids)
        if self.positions is not None:
            positions = torch.arange(ids.shape[1], device=ids.device)
            x = x + self.positions(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.config.gradient_checkpointing else block(x)
        return F.linear(self.final_norm(x), self.tokens.weight)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def config_dict(self):
        return asdict(self.config)


def token_losses(logits, labels):
    """Exactly one causal shift; returns per-target NLL and supervision mask."""
    targets = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                            ignore_index=-100, reduction='none')
    return losses, targets.ne(-100)
