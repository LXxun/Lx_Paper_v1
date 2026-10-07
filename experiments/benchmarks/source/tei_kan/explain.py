"""Exact three-player Shapley and coalition-weighted temporal path integrals.

These are baseline-dependent model attributions, not raw-input causal effects.
"""
import math
import torch
from .model import PAIRS


def fixed_target(scores, kind="margin"):
    if kind not in {"margin", "raw_intensity", "intensity", "probability"}:
        raise ValueError("Unknown explanation target")
    ranks = scores[:, :3].argsort(dim=1, descending=True)
    weights = torch.zeros_like(scores)
    if kind in {"margin", "probability"}:
        weights.scatter_(1, ranks[:, :1], 1)
        if kind == "margin":
            weights.scatter_(1, ranks[:, 1:2], -1)
    else:
        weights[:, 3] = 1
    return {"kind": kind, "weights": weights.detach(), "winner": ranks[:, 0], "runner_up": ranks[:, 1]}


def target_values(scores, kind, weights):
    if kind == "intensity":
        return 3 * torch.tanh(scores[..., 3] / 3)
    if kind == "probability":
        return (scores[..., :3].softmax(-1) * weights[..., :3]).sum(-1)
    return (scores * weights).sum(-1)


@torch.no_grad()
def exact_shapley(model, z, target):
    if model.training:
        raise ValueError("Explanation requires model.eval()")
    membership = torch.tensor([[(s >> m) & 1 for m in range(3)] for s in range(8)], device=z.device, dtype=z.dtype)
    masked = z[:, None] * membership[None, :, :, None]
    scores = model.fuse(masked.flatten(0, 1))["scores"].reshape(len(z), 8, 4)
    values = target_values(scores, target["kind"], target["weights"][:, None])
    phi = torch.zeros((len(z), 3), device=z.device, dtype=z.dtype)
    for m in range(3):
        for subset in range(8):
            if subset & (1 << m):
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(2 - size) / 6
            phi[:, m] += weight * (values[:, subset | (1 << m)] - values[:, subset])
    out = {"phi": phi, "coalition_values": values, "baseline": values[:, 0], "value": values[:, 7],
           "shapley_residual": phi.sum(1) - (values[:, 7] - values[:, 0])}
    fused = model.fuse(z)
    if fused["main"] is not None and target["kind"] in {"margin", "raw_intensity"}:
        main = (fused["main"] * target["weights"][:, None]).sum(-1)
        pair = (fused["interactions"] * target["weights"][:, None]).sum(-1)
        closed = main.clone()
        for j, (m, n) in enumerate(PAIRS):
            closed[:, m] += pair[:, j] / 2
            closed[:, n] += pair[:, j] / 2
        out.update(main=main, interactions=pair, closed_form_phi=closed, closed_form_residual=closed - phi)
    return out


def temporal_integral(model, evidence, target, steps=32):
    if steps < 1 or model.training:
        raise ValueError("Positive integration steps and eval mode are required")
    # Upstream attention/encoders are held at the original input. The integration
    # path is in pooled-evidence space, not the raw token/frame input space.
    evidence = evidence.detach()
    z = evidence.sum(2)
    batch, _, width = z.shape
    result = torch.zeros(evidence.shape[:3], device=z.device, dtype=z.dtype)
    alphas = torch.linspace(0, 1, steps + 1, device=z.device, dtype=z.dtype)
    with torch.enable_grad():
        for m in range(3):
            integrated = torch.zeros_like(z[:, m])
            for subset in range(8):
                if subset & (1 << m):
                    continue
                size = subset.bit_count()
                weight = math.factorial(size) * math.factorial(2 - size) / 6
                membership = z.new_tensor([(subset >> j) & 1 for j in range(3)])
                total_gradient = torch.zeros_like(integrated)
                # Bound working memory independently of the requested quadrature resolution.
                for begin in range(0, len(alphas), 65):
                    alpha = alphas[begin:begin + 65]
                    path = (z[:, None] * membership[None, None, :, None]).expand(batch, len(alpha), 3, width).clone()
                    path[:, :, m] = alpha[None, :, None] * z[:, None, m]
                    flat = path.reshape(-1, 3, width).detach().requires_grad_(True)
                    scores = model.fuse(flat)["scores"]
                    weights = target["weights"][:, None].expand(batch, len(alpha), 4).reshape(-1, 4)
                    values = target_values(scores, target["kind"], weights)
                    gradients = torch.autograd.grad(values.sum(), flat)[0].reshape(batch, len(alpha), 3, width)[:, :, m]
                    trapezoid = torch.ones_like(alpha)
                    if begin == 0:
                        trapezoid[0] = 0.5
                    if begin + len(alpha) == len(alphas):
                        trapezoid[-1] = 0.5
                    total_gradient += (gradients * trapezoid[None, :, None]).sum(1).detach() / steps
                integrated += weight * total_gradient
            result[:, m] = (evidence[:, m] * integrated[:, None]).sum(-1)
    return result


def explain_output(model, out, kind="margin", steps=32, max_steps=512, atol=1e-4, rtol=1e-3):
    target = fixed_target(out["scores"].detach(), kind)
    shapley = exact_shapley(model, out["z"].detach(), target)
    used = steps
    while True:
        local = temporal_integral(model, out["evidence"], target, used)
        residual = local.sum(-1) - shapley["phi"]
        converged = residual.abs() <= (atol + rtol * shapley["phi"].abs())
        if bool(converged.all()) or used >= max_steps:
            break
        used = min(used * 2, max_steps)
    return dict(shapley, target=target, local=local, local_residual=residual,
                integration_steps=used, integration_converged=converged)
