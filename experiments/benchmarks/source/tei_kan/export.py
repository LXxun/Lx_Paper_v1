"""CSV exports of predictions, signed effects and feature-position evidence."""
import csv
from pathlib import Path
import numpy as np
import torch
from .explain import explain_output, exact_shapley, fixed_target


def write_csv(rows, path):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dominant_modality(phi):
    phi = np.asarray(phi)
    if phi.max() <= 1e-8:
        return "no_positive_support"
    names = [name for name, value in zip(("T", "A", "V"), phi) if np.isclose(value, phi.max(), atol=1e-7, rtol=1e-5)]
    return "+".join(names)


def token_mapping(tokenizer, tokens, raw_text=None):
    pieces = tokenizer.convert_ids_to_tokens(tokens[0].tolist()) if tokenizer else [str(int(x)) for x in tokens[0]]
    offsets = None
    status = "token_position_only"
    if tokenizer and raw_text:
        encoded = tokenizer(raw_text, max_length=50, padding="max_length", truncation=True, return_offsets_mapping=True)
        if np.array_equal(encoded["input_ids"], tokens[0]):
            offsets = encoded["offset_mapping"]
            status = "raw_text_offsets_verified"
        else:
            status = "raw_text_tokenization_mismatch"
    return pieces, offsets, status


def export_cached(model, data, output, local=True, tokenizer=None, batch_size=8, steps=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions, components, evidence = [], [], []
    max_shapley, max_local, unconverged = 0.0, 0.0, 0
    model.eval()
    for start in range(0, data.n, batch_size):
        idx = torch.arange(start, min(start + batch_size, data.n), device=data.device)
        with torch.no_grad():
            out = model(*data.inputs(idx))
        if local:
            cls = explain_output(model, out, "margin", steps=steps)
        else:
            cls = exact_shapley(model, out["z"], fixed_target(out["scores"]))
        strength = exact_shapley(model, out["z"], fixed_target(out["scores"], "intensity"))
        prob = out["logits"].softmax(-1).cpu().numpy()
        raw_phi, reg_phi = cls["phi"].cpu().numpy(), strength["phi"].cpu().numpy()
        max_shapley = max(max_shapley, float(cls["shapley_residual"].abs().max()), float(strength["shapley_residual"].abs().max()))
        if local:
            max_local = max(max_local, float(cls["local_residual"].abs().max()))
            unconverged += int((~cls["integration_converged"].all(-1)).sum())
        for j in range(len(idx)):
            row_index = start + j
            sample_id = str(data.base["ids"][row_index])
            label = int(prob[j].argmax())
            magnitude = np.abs(raw_phi[j]).sum()
            row = {"sample_id": sample_id, "pred_label": label, "pred_name": ["Negative", "Neutral", "Positive"][label],
                   "pred_intensity": float(out["reg"][j]), "p_negative": float(prob[j, 0]), "p_neutral": float(prob[j, 1]), "p_positive": float(prob[j, 2]),
                   "class_explanation_target": "fixed_winner_minus_runner_up_logit", "primary_modality": dominant_modality(raw_phi[j]),
                   "class_baseline": float(cls["baseline"][j]), "class_target_value": float(cls["value"][j]),
                   "intensity_baseline": float(strength["baseline"][j]),
                   "zero_observed_modalities": "+".join(name for m, name in enumerate(("T", "A", "V")) if not bool(out["available"][j, m]))}
            for m, name in enumerate(("T", "A", "V")):
                row["class_phi_" + name] = float(raw_phi[j, m])
                row["intensity_phi_" + name] = float(reg_phi[j, m])
                row["relative_absolute_effect_" + name] = float(abs(raw_phi[j, m]) / magnitude) if magnitude > 1e-8 else ""
            row["class_shapley_residual"] = float(cls["shapley_residual"][j])
            row["local_integration_converged"] = bool(cls["integration_converged"][j].all()) if local else "not_requested"
            predictions.append(row)
            if out["main"] is not None:
                weights = fixed_target(out["scores"])["weights"][j]
                for name, vector in [("bias", model.bias)] + [("U_" + name, out["main"][j, m]) for m, name in enumerate(("T", "A", "V"))] + [("I_" + name, out["interactions"][j, m]) for m, name in enumerate(("TA", "TV", "AV"))]:
                    vector = vector.detach()
                    components.append({"sample_id": sample_id, "component": name, "negative_logit": float(vector[0]),
                                       "neutral_logit": float(vector[1]), "positive_logit": float(vector[2]),
                                       "raw_intensity_score": float(vector[3]), "class_margin_effect": float((vector * weights).sum()),
                                       "space": "logits_and_pre_tanh_intensity"})
            if not local:
                continue
            original_text = str(data.base["raw_text"][row_index]) if "raw_text" in data.base else None
            pieces, offsets, alignment = token_mapping(tokenizer, data.base["tokens"][row_index], original_text)
            for m, name in enumerate(("T", "A", "V")):
                positions = np.flatnonzero(data.base["mask"][row_index, :, m])
                for t in positions:
                    character_start = offsets[t][0] if m == 0 and offsets is not None else ""
                    character_end = offsets[t][1] if m == 0 and offsets is not None else ""
                    evidence.append({"sample_id": sample_id, "modality": name, "position": int(t),
                                     "token_id": int(data.base["tokens"][row_index, 0, t]) if m == 0 else "",
                                     "token_piece": pieces[t] if m == 0 else "", "character_start": character_start, "character_end": character_end,
                                     "text_excerpt": original_text[character_start:character_end] if isinstance(character_start, int) else "",
                                     "signed_class_margin_contribution": float(cls["local"][j, m, t]),
                                     "start_s": "", "end_s": "", "frame_file": "",
                                     "alignment_status": alignment if m == 0 else "feature_position_only_timestamps_unverified",
                                     "attribution_space": "contextual_pooled_evidence", "integration_steps": cls["integration_steps"]})
    write_csv(predictions, output / "predictions.csv")
    write_csv(components, output / "interactions.csv")
    write_csv(evidence, output / "evidence.csv")
    return {"samples": data.n, "prediction_rows": len(predictions), "component_rows": len(components), "evidence_rows": len(evidence),
            "max_shapley_residual": max_shapley, "max_local_residual": max_local if local else None,
            "local_integration_unconverged_samples": unconverged, "timestamps": "not_inferred_from_sequence_indices"}
