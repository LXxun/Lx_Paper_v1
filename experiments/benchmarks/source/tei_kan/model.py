"""Temporal evidence bottleneck and anchored, explicitly additive KAN fusion."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from .kan import SmallKAN

MODALITIES = ("T", "A", "V")
PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass
class ModelConfig:
    input_dims: tuple = (768, 74, 35)
    hidden_dim: int = 64
    bottleneck_dim: int = 8
    kan_hidden_dim: int = 8
    dropout: float = 0.1
    dilations: tuple = (1, 2)
    spline_degree: int = 3
    grid_intervals: int = 5
    latent_bound: float = 3.0
    interaction: bool = True
    temporal: bool = True
    evidence_pool: bool = True
    fusion: str = "structured_kan"

    def __post_init__(self):
        if len(self.input_dims) != 3 or min(self.input_dims) <= 0:
            raise ValueError("Three positive modality dimensions are required")
        if self.fusion not in {"structured_kan", "concat_kan", "structured_mlp"}:
            raise ValueError("Unknown fusion architecture")


def masked_softmax(scores, mask):
    weights = torch.softmax(scores.float().masked_fill(~mask, -1e9), dim=1) * mask
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-12)


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation, groups=hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, mask):
        # LayerNorm bias at a padded position must not enter the next convolution.
        q = self.norm(h) * mask.unsqueeze(-1)
        q = self.conv(q.transpose(1, 2)).transpose(1, 2)
        return (h + self.drop(self.ff(q))) * mask.unsqueeze(-1)


class EvidenceEncoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        h, k = config.hidden_dim, config.bottleneck_dim
        self.project = nn.Sequential(nn.Linear(input_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(config.dropout))
        self.temporal = nn.ModuleList([TemporalBlock(h, d, config.dropout) for d in config.dilations] if config.temporal else [])
        self.bottleneck = nn.Sequential(nn.Linear(h, 32), nn.GELU())
        self.mu = nn.Linear(32, k)
        self.rho = nn.Linear(32, k)
        nn.init.constant_(self.rho.bias, -2.0)
        self.attention = nn.Sequential(nn.Linear(h, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x, mask):
        h = self.project(x) * mask.unsqueeze(-1)
        for block in self.temporal:
            h = block(h, mask)
        q = self.bottleneck(h)
        mu = self.mu(q)
        sigma = F.softplus(self.rho(q)) + 1e-4
        latent = mu + sigma * torch.randn_like(mu) if self.training else mu
        latent = self.config.latent_bound * torch.tanh(latent / self.config.latent_bound)
        score = self.attention(h).squeeze(-1) if self.config.evidence_pool else torch.zeros_like(mask, dtype=x.dtype)
        weights = masked_softmax(score, mask)
        evidence = weights.unsqueeze(-1) * latent
        kl = 0.5 * (mu.square() + sigma.square() - 2 * sigma.log() - 1).mean(-1)
        kl = (kl * mask).sum(1) / mask.sum(1).clamp_min(1)
        return {"pooled": evidence.sum(1), "evidence": evidence, "attention": weights,
                "kl": kl, "available": mask.any(1)}


class TEIKAN(nn.Module):
    def __init__(self, config=None, stats=None):
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**(config or {}))
        c = self.config
        for name, dim in zip(("text", "audio", "vision"), c.input_dims):
            mean = torch.zeros(dim) if stats is None else torch.as_tensor(stats[name + "_mean"], dtype=torch.float32)
            std = torch.ones(dim) if stats is None else torch.as_tensor(stats[name + "_std"], dtype=torch.float32)
            if mean.shape != (dim,) or std.shape != (dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Invalid training normalization statistics")
            self.register_buffer(name + "_mean", mean.clone())
            self.register_buffer(name + "_std", std.clone())
        self.encoders = nn.ModuleList([EvidenceEncoder(d, c) for d in c.input_dims])
        self.auxiliary = nn.ModuleList([nn.Linear(c.bottleneck_dim, 4) for _ in range(3)])
        def branch(width):
            if c.fusion == "structured_mlp":
                # Width chosen to roughly match the parameter count of a KAN branch.
                return nn.Sequential(nn.Linear(width, 8 * c.kan_hidden_dim), nn.Tanh(), nn.Linear(8 * c.kan_hidden_dim, 4))
            return SmallKAN(width, c.kan_hidden_dim, 4, c.grid_intervals, c.spline_degree, c.latent_bound)
        if c.fusion == "concat_kan":
            self.joint = branch(c.bottleneck_dim * 3)
            self.unimodal = nn.ModuleList()
            self.pairwise = nn.ModuleList()
        else:
            self.unimodal = nn.ModuleList([branch(c.bottleneck_dim) for _ in range(3)])
            self.pairwise = nn.ModuleList([branch(c.bottleneck_dim * 2) for _ in PAIRS] if c.interaction else [])
        self.bias = nn.Parameter(torch.zeros(4))

    def config_dict(self):
        return asdict(self.config)

    def fuse(self, z):
        if z.ndim != 3 or z.shape[1:] != (3, self.config.bottleneck_dim):
            raise ValueError("Expected latent tensor [batch,3,bottleneck_dim]")
        zero = torch.zeros_like(z[:, 0])
        if self.config.fusion == "concat_kan":
            scores = self.bias + self.joint(z.flatten(1)) - self.joint(torch.zeros_like(z).flatten(1))
            return {"scores": scores, "main": None, "interactions": None}
        main = torch.stack([f(z[:, m]) - f(zero) for m, f in enumerate(self.unimodal)], dim=1)
        interactions = []
        for f, (m, n) in zip(self.pairwise, PAIRS):
            # Batched evaluation also ensures all four anchor terms share the same operations.
            inputs = torch.cat([torch.cat([z[:, m], z[:, n]], -1), torch.cat([z[:, m], zero], -1),
                                torch.cat([zero, z[:, n]], -1), torch.cat([zero, zero], -1)], 0)
            both, left, right, origin = f(inputs).chunk(4, 0)
            interactions.append(both - left - right + origin)
        pair = torch.stack(interactions, 1) if interactions else z.new_zeros((len(z), 3, 4))
        return {"scores": self.bias + main.sum(1) + pair.sum(1), "main": main, "interactions": pair}

    def forward(self, text, audio, vision, mask):
        if mask.dtype != torch.bool or mask.ndim != 3 or mask.shape[-1] != 3:
            raise ValueError("Expected boolean observation mask [batch,time,3]")
        encoded = []
        for m, (name, x, encoder) in enumerate(zip(("text", "audio", "vision"), (text, audio, vision), self.encoders)):
            if x.shape[:2] != mask.shape[:2] or x.shape[-1] != self.config.input_dims[m]:
                raise ValueError("Feature shape does not match model and mask")
            x = (x.float() - getattr(self, name + "_mean")) / getattr(self, name + "_std")
            # Text uses identity normalization; A/V use train-only statistics.
            if m:
                x = x.clamp(-8, 8)
            x = torch.where(mask[:, :, m, None], x, torch.zeros_like(x))
            encoded.append(encoder(x, mask[:, :, m]))
        z = torch.stack([e["pooled"] for e in encoded], 1)
        out = self.fuse(z)
        aux = torch.stack([head(z[:, m]) for m, head in enumerate(self.auxiliary)], 1)
        out.update({"logits": out["scores"][:, :3], "reg": 3 * torch.tanh(out["scores"][:, 3] / 3),
                    "z": z, "evidence": torch.stack([e["evidence"] for e in encoded], 1),
                    "attention": torch.stack([e["attention"] for e in encoded], 1),
                    "kl": torch.stack([e["kl"] for e in encoded], 1),
                    "available": torch.stack([e["available"] for e in encoded], 1),
                    "aux_logits": aux[:, :, :3], "aux_reg": 3 * torch.tanh(aux[:, :, 3] / 3)})
        return out
