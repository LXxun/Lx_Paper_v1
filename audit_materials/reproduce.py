"""Rebuild eight numeric tables and 14 adjusted intervals; no model inference.

Requires Python >=3.9 and NumPy. Run from any directory with --out NEW_DIRECTORY.
"""
from pathlib import Path
import argparse, hashlib, json, platform, subprocess, sys
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def bootstrap(family, dataset, rows):
    # Row order is the preserved original cluster order, with identifiers removed.
    counts = np.array([r["n"] for r in rows], dtype=np.int64)
    sums = np.array([r["sums"] for r in rows], dtype=np.float64)
    assert len({r["group"] for r in rows}) == len(rows)
    assert (counts > 0).all() and np.isfinite(sums).all()
    seed = {"original": {"MOSI": 2026100201, "MOSEI": 2026100202},
            "recurrent": {"MOSI": 2026100701, "MOSEI": 2026100702},
            "mult": {"MOSI": 20261005, "MOSEI": 20261005}}[family][dataset]
    B = 20000 if family == "mult" else 50000
    batch = 500 if family == "original" else 1000
    alpha = .05 / 12 if family == "original" else .00625
    rng = np.random.default_rng(seed)
    if family == "mult":
        values = sums / counts[:, None]
        point = values.mean(0)
    else:
        point = sums.sum(0) / counts.sum()
    draws = []
    for lo in range(0, B, batch):
        ix = rng.integers(0, len(rows), size=(min(batch, B-lo), len(rows)))
        if family == "mult":
            draws.append(values[ix].mean(1))
        else:
            draws.append(sums[ix].sum(1) / counts[ix].sum(1)[:, None])
    boot = np.concatenate(draws)
    ci = np.quantile(boot, [alpha, 1-alpha], axis=0, method="linear")
    result = []
    for j, name in enumerate(["delete_matching", "retain_matching", "reference_vs_target"][:3 if family == "original" else 2]):
        result.append(dict(family=family, dataset=dataset, contrast=name,
                           mean=float(point[j]), adjusted_ci=ci[:, j].tolist(),
                           samples=int(counts.sum()), videos=len(rows),
                           replicates=B, random_seed=seed,
                           weighting="equal-video" if family == "mult" else "equal-segment"))
    return result

def expected(family, dataset, contrast):
    if family == "original":
        q = next(q for q in read("data/final_test/primary_six.json") if q["id"] == dataset+"_"+contrast)
        return [q["mean"], *q["adjusted_ci"]]
    op = "delete" if contrast == "delete_matching" else "retain"
    if family == "recurrent":
        q = next(q for q in read("data/recurrent_extension/RESULTS.json")["summary"][dataset]["primary"] if q["operation"] == op)
        return [q["mean"], q["ci_lower"], q["ci_upper"]]
    q = next(q for q in read("data/mult_space_validation/ANALYSIS.json")["summary"] if q["dataset"] == dataset and q["primary"] and q["operation"] == op)
    return [q["mean"], *q["ci98_75_bonferroni4"]]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        ap.error("Use a directory outside audit_materials.")
    if out.exists():
        ap.error("Use a new output directory.")
    pin = read("MANIFEST.json")
    for rel, digest in pin["files"].items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError("Input or implementation changed: " + rel)
    out.mkdir(parents=True)
    for script, folder in [("reproduce_main_tables.py", "main_tables"), ("reproduce_recurrent_tables.py", "recurrent_tables")]:
        subprocess.run([sys.executable, str(ROOT / "source" / script), "--out", str(out / folder), "--check"], check=True)
    results = []
    for family in ["original", "recurrent", "mult"]:
        source = read("clusters/"+family+".json")
        for ds, rows in source["datasets"].items():
            for result in bootstrap(family, ds, rows):
                target = expected(family, ds, result["contrast"])
                actual = [result["mean"], *result["adjusted_ci"]]
                error = float(np.max(np.abs(np.array(actual)-np.array(target))))
                assert error < 1e-12, (family, ds, result["contrast"], error)
                result.update(reference=target, max_absolute_error=error, passed=True)
                results.append(result)
    assert len(results) == 14
    report = dict(status="passed", numeric_tables=8, numeric_rows=36,
                  adjusted_intervals_recomputed=14, results=results,
                  python=platform.python_version(), numpy=np.__version__,
                  inference_run=False, original_records_reconstructed=False,
                  scope="Eight tables from archived aggregates; fourteen intervals from anonymous video sums/counts. Does not validate model inference or reconstruct individual selections.")
    (out / "AUDIT_REPORT.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("PASS: 8 tables / 36 rows; 14 adjusted intervals recomputed; no model inference.")

if __name__ == "__main__":
    main()
