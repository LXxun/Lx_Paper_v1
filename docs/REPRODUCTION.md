# Using the published code

## Synthetic verification

`python tests/test_core.py` requires NumPy and PyTorch only. It loads no participant data or trained checkpoints and covers grouping, negative-score/earliest-tie selection, evidence pooling and the bounded regression output.

## Training inputs and configuration

Obtain the MMSA MOSI/MOSEI aligned feature packages separately under their providers' terms. Place them at `experiments/benchmarks/data/{MOSI,MOSEI}/aligned_50.pkl`. The preparation script checks fixed dataset hashes and video folds; obtain CMU Multimodal SDK under `experiments/benchmarks/repos/CMU-MultimodalSDK`. Set the environment variable `BERT_MODEL_DIR` to a local bert-base-uncased model/tokenizer directory. Third-party baseline experiments additionally need the upstream MMSA repository under `experiments/benchmarks/repos/MMSA`.

After provisioning those dependencies, for example:

```sh
cd experiments/benchmarks
python prepare_standard.py --dataset MOSI
python run_compact_stage.py --model tei_mlp --variant main_only --dataset MOSI --seed 2026 --gpu 0
```

The guarded final-test and recurrent follow-up runners additionally require checkpoints, manifests and locks. The historical execution locks are not in this release. Do not fabricate or bypass them: audit the exact assets and implementation for your own run and record new hashes. Public portability changes alter code hashes from the historical execution. `docs/SOURCE_PROVENANCE.json` records the fetched source hashes and public export hashes.

## Table and inference reconstruction

The complete lightweight numerical entry is now:

```sh
python audit_materials/reproduce.py --out outputs/numerical_audit
```

The bundled inputs rebuild eight tables (36 rows), the reported hybrid aggregate magnitudes, and fourteen adjusted intervals. The script verifies hashes, executes table checks, then recalculates the bootstrap from anonymous video sums/counts. It reports effective samples, clusters, seeds, weights and differences from the archived reference values. It needs only Python and NumPy and performs no inference. The [bundle README](../audit_materials/README.md) documents scope and exact commands.

The older root-level `source/reproduce_*` and `source/build_heldout_matrix.py` entries retain their original archive requirements for historical inspection. Use the self-contained `audit_materials` entry for the public numerical audit. Checkpoints, raw records and features are still required to verify the upstream generation of those statistics.

## Interpretation

The original six contrasts and four recurrent follow-up contrasts are separate families. The latter reused the original test cohort after inspecting the original results, shares the tokenizer/evidence head and changes pretraining. Its implementation clarification and numerical amendment document bounded-output selection and the uniform FP32 restart; no claim of pristine prospective preregistration or attribution-method superiority is made.
