"""Question-3 data contract. No missing-data augmentation or test-based statistics."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

NAMES = ("text", "audio", "vision")


def load_supplied_pickle(path):
    """Read trusted contest files serialized by NumPy 1.x or 2.x.

    NumPy 2 moved pickle module names to numpy._core. Only that namespace
    is remapped on NumPy 1; array values/shapes and the source file are unchanged.
    """
    import pickle
    class ContestUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if int(np.__version__.split('.')[0]) < 2 and module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
            return super().find_class(module, name)
    with Path(path).open('rb') as f:
        return ContestUnpickler(f).load()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verify_encoder(model_path, expected_hash):
    weight = Path(model_path) / "model.safetensors"
    if not weight.exists():
        weight = Path(model_path) / "pytorch_model.bin"
    if not expected_hash or sha256(weight) != expected_hash:
        raise ValueError("Frozen BERT weights do not match the training interface")


def interface(raw, labels=True, sample_ids=None):
    t = np.asarray(raw["text_bert"])
    if not np.isfinite(t).all() or not (t == np.round(t)).all():
        raise ValueError("Non-integer BERT input")
    t = t.astype(np.int64).reshape(-1, 3, 50)
    a = np.asarray(raw["audio"], np.float32).reshape(-1, 50, 74)
    v = np.asarray(raw["vision"], np.float32).reshape(-1, 50, 35)
    n = len(t)
    if len(a) != n or len(v) != n or not np.isfinite(a).all() or not np.isfinite(v).all():
        raise ValueError("Invalid feature shapes or nonfinite values")
    if not np.isin(t[:, 1], [0, 1]).all() or not np.isin(t[:, 2], [0, 1]).all() or t[:, 0].min() < 0 or t[:, 0].max() >= 30522:
        raise ValueError("Invalid bert-base-uncased token interface")
    text_mask = (t[:, 1] == 1) & ~np.isin(t[:, 0], [0, 101, 102])
    # An all-zero A/V row is not affirmative evidence. Exclude it from pooling,
    # but do not label it as known padding or synthetic missingness.
    audio_mask = np.any(a != 0, axis=-1)
    vision_mask = np.any(v != 0, axis=-1)
    mask = np.stack([text_mask, audio_mask, vision_mask], axis=-1)
    if sample_ids is None:
        if "id" not in raw:
            raise ValueError("Sample IDs are required")
        sample_ids = np.asarray(raw["id"]).reshape(-1)
    ids = np.asarray(sample_ids).astype(str).reshape(-1)
    if len(ids) != n or len(set(ids.tolist())) != n:
        raise ValueError("IDs must be unique within each split")
    out = {"tokens": t, "audio": a, "vision": v, "mask": mask, "ids": ids}
    if labels:
        raw_yc = np.asarray(raw["classification_labels"]).reshape(-1)
        yc = raw_yc.astype(np.int64)
        yr = np.asarray(raw["regression_labels"], np.float32).reshape(-1)
        if len(yc) != n or len(yr) != n or not np.isfinite(yr).all() or not np.isin(yc, [0, 1, 2]).all():
            raise ValueError("Invalid sentiment labels")
        if not np.array_equal(yc, raw_yc) or not np.array_equal(yc, np.sign(yr) + 1) or np.any(np.abs(yr) > 3):
            raise ValueError("Classification and original regression labels disagree")
        out.update(yc=yc, yr=yr)
    if "raw_text" in raw:
        texts = np.asarray(raw["raw_text"]).astype(str).reshape(-1)
        if len(texts) != n:
            raise ValueError("Raw-text count differs from feature count")
        out["raw_text"] = texts
    return out


def fit_stats(train):
    stats = {"text_mean": np.zeros(768, np.float32), "text_std": np.ones(768, np.float32)}
    for m, name in [(1, "audio"), (2, "vision")]:
        rows = train[name][train["mask"][:, :, m]].astype(np.float64)
        if not len(rows):
            raise ValueError("Cannot fit statistics without observed training rows")
        mean, std = rows.mean(0), rows.std(0)
        std[std < 1e-3] = 1
        stats[name + "_mean"] = mean.astype(np.float32)
        stats[name + "_std"] = std.astype(np.float32)
    return stats


class FrozenBert:
    def __init__(self, model_path, device="cuda:0", batch=128):
        from transformers import BertModel
        self.device, self.batch = torch.device(device), batch
        self.model = BertModel.from_pretrained(model_path, local_files_only=True, attn_implementation="sdpa").eval().to(self.device)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, tokens, text_mask):
        out = np.zeros((len(tokens), 50, 768), dtype=np.float16)
        for start in range(0, len(tokens), self.batch):
            t = torch.as_tensor(tokens[start:start + self.batch], device=self.device, dtype=torch.long)
            am = t[:, 1].clone()
            am[am.sum(1) == 0, 0] = 1
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                h = self.model(input_ids=t[:, 0], attention_mask=am, token_type_ids=t[:, 2]).last_hidden_state
            values = h.float().cpu().numpy() * text_mask[start:start + self.batch, :, None]
            out[start:start + self.batch] = values.astype(np.float16)
        return out


class CachedSplit:
    def __init__(self, cache, split, device="cpu", limit=None):
        cache = Path(cache)
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("normalization_fit_split") != "train" or manifest.get("uses_augmentation") is not False:
            raise ValueError("Expected an independently audited clean Q3 cache")
        raw = dict(np.load(cache / f"{split}_base.npz", allow_pickle=False))
        text = np.load(cache / f"{split}_text.npy", mmap_mode="r")
        if text.ndim != 3 or text.shape[1:] != (50, 768):
            raise ValueError("Only clean text [N,50,768] is accepted, not augmentation banks")
        n = len(raw["ids"]) if limit is None else min(limit, len(raw["ids"]))
        if n <= 0 or len(text) != len(raw["ids"]):
            raise ValueError("Invalid cache sample count")
        self.base = {k: v[:n] for k, v in raw.items()}
        self.n, self.device = n, torch.device(device)
        self.tensors = {k: torch.as_tensor(self.base[k], device=self.device) for k in ("audio", "vision", "mask")}
        self.tensors["text"] = torch.as_tensor(np.array(text[:n]), device=self.device)
        for key in ("yc", "yr"):
            if key in self.base:
                self.tensors[key] = torch.as_tensor(self.base[key], device=self.device)

    def inputs(self, indices):
        return tuple(self.tensors[k][indices] for k in ("text", "audio", "vision", "mask"))


def prediction_metrics(yc, yr, probability, intensity):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix, classification_report
    label = np.asarray(probability).argmax(1)
    corr = float(np.corrcoef(yr, intensity)[0, 1]) if np.std(yr) > 1e-8 and np.std(intensity) > 1e-8 else None
    return {"accuracy": float(accuracy_score(yc, label)),
            "macro_f1": float(f1_score(yc, label, labels=[0, 1, 2], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yc, label, labels=[0, 1, 2], average="weighted", zero_division=0)),
            "mae": float(mean_absolute_error(yr, intensity)), "pearson": corr,
            "confusion_matrix": confusion_matrix(yc, label, labels=[0, 1, 2]).tolist(),
            "per_class": classification_report(yc, label, labels=[0, 1, 2], target_names=["Negative", "Neutral", "Positive"], output_dict=True, zero_division=0),
            "polarity_strength_disagreement": float(np.mean(label != np.sign(intensity) + 1))}
