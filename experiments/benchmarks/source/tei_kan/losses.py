"""Joint prediction loss and optional simplified Pareto encoder update."""
import torch
from torch.nn import functional as F


def loss_components(out, yc, yr, regression_weight=0.75, auxiliary_weight=0.1, kl_weight=1e-4, huber_delta=0.5):
    task = F.cross_entropy(out["logits"], yc) + regression_weight * F.huber_loss(out["reg"], yr, delta=huber_delta)
    auxiliary = []
    for m in range(3):
        values = F.cross_entropy(out["aux_logits"][:, m], yc, reduction="none")
        values = values + regression_weight * F.huber_loss(out["aux_reg"][:, m], yr, delta=huber_delta, reduction="none")
        available = out["available"][:, m]
        auxiliary.append((values * available).sum() / available.sum().clamp_min(1))
    aux = torch.stack(auxiliary).mean()
    kl = out["kl"].mean()
    return {"task": task, "auxiliary": aux, "kl": kl,
            "per_modality_aux": auxiliary, "auxiliary_weight": auxiliary_weight, "kl_weight": kl_weight,
            "total": task + auxiliary_weight * aux + kl_weight * kl}


def _grads(loss, params):
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, values)]


def backward_with_pareto(model, losses):
    replacements, conflicts = [], []
    for m, encoder in enumerate(model.encoders):
        params = list(encoder.parameters())
        gf = _grads(losses["task"], params)
        ga = _grads(losses["per_modality_aux"][m] * (losses["auxiliary_weight"] / 3), params)
        gk = _grads(losses["kl"] * losses["kl_weight"], params)
        dot = sum((f * a).sum() for f, a in zip(gf, ga))
        if dot.item() < 0:
            aa = sum(a.square().sum() for a in ga)
            denominator = sum((f - a).square().sum() for f, a in zip(gf, ga))
            alpha = ((aa - dot) / denominator.clamp_min(1e-20)).clamp(0, 1)
            gradients = [2 * (alpha * f + (1 - alpha) * a) + k for f, a, k in zip(gf, ga, gk)]
            conflicts.append(m)
        else:
            gradients = [f + a + k for f, a, k in zip(gf, ga, gk)]
        replacements.extend(zip(params, gradients))
    losses["total"].backward()
    for parameter, gradient in replacements:
        parameter.grad = gradient.detach()
    return conflicts
