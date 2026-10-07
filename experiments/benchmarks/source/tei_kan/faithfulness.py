"""Contiguous-unit perturbations. Other modalities remain fixed in each trial."""
import numpy as np
import torch


def evidence_units(mask, token_ids=None, tokenizer=None):
    positions = np.flatnonzero(mask)
    if token_ids is None:
        return [[int(t)] for t in positions]
    if tokenizer is None:
        raise ValueError("A tokenizer is required to perturb whole WordPiece words")
    pieces = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    groups = []
    for t in positions:
        if pieces[t].startswith("##") and groups and groups[-1][-1] == t - 1:
            groups[-1].append(int(t))
        else:
            groups.append([int(t)])
    return groups


def select_span(units, scores, fraction, rng=None):
    if not units:
        return []
    count = min(len(units), max(1, int(np.ceil(len(units) * fraction))))
    if rng is None:
        values = np.asarray([sum(float(scores[t]) for t in unit) for unit in units])
        totals = np.convolve(values, np.ones(count), mode="valid")
        start = int(totals.argmax())
    else:
        start = int(rng.integers(len(units) - count + 1))
    return [position for unit in units[start:start + count] for position in unit]


def perturb_inputs(model, original, tokens, modality, positions, retain, encoder=None):
    values = [x.clone() for x in original]
    mask = values[3][0, :, modality].detach().cpu().numpy()
    selected = np.zeros(len(mask), dtype=bool); selected[positions] = True
    replace = mask & (~selected if retain else selected)
    if modality == 0:
        if encoder is None:
            raise ValueError("Text must be replaced before BERT and re-encoded")
        changed_tokens = tokens.copy()
        changed_tokens[0, 0, replace] = 103  # [MASK], attention and sequence length retained.
        encoded = encoder.encode(changed_tokens, values[3][:, :, 0].cpu().numpy())
        values[0] = torch.as_tensor(encoded, device=values[0].device)
    else:
        reference = getattr(model, ("audio", "vision")[modality - 1] + "_mean")
        indices = torch.as_tensor(np.flatnonzero(replace), device=values[modality].device)
        values[modality][0, indices] = reference
    # Keep observation masks fixed: the intervention is reference substitution,
    # not training-style missing-modality masking.
    return tuple(values)
