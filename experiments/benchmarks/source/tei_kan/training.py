"""Training helpers with validation-only checkpoint selection."""
from pathlib import Path
import torch
import numpy as np
from .model import TEIKAN
from .data import prediction_metrics


def make_optimizer(model, encoder_lr=3e-4, kan_lr=5e-4, weight_decay=1e-4):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        fusion = name.startswith(("unimodal.", "pairwise.", "joint.")) or name == "bias"
        decay = parameter.ndim >= 2 and "norm" not in name.lower()
        key = (kan_lr if fusion else encoder_lr, weight_decay if decay else 0.0)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([{"params": p, "lr": lr, "weight_decay": wd} for (lr, wd), p in groups.items()])


@torch.inference_mode()
def predict(model, data, batch_size=128):
    model.eval()
    probability, intensity = [], []
    for start in range(0, data.n, batch_size):
        indices = torch.arange(start, min(start + batch_size, data.n), device=data.device)
        out = model(*data.inputs(indices))
        probability.append(out["logits"].softmax(-1).cpu().numpy())
        intensity.append(out["reg"].cpu().numpy())
    return np.concatenate(probability), np.concatenate(intensity)


def evaluate(model, data, batch_size=128):
    probability, intensity = predict(model, data, batch_size)
    report = prediction_metrics(data.base["yc"], data.base["yr"], probability, intensity)
    report["selection_score"] = 0.5 * report["macro_f1"] + 0.5 * (1 - report["mae"] / 6)
    return report, probability, intensity


def load_checkpoint(path, device="cpu"):
    # This loader accepts only our locally produced tensor/config checkpoints.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = TEIKAN(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint
