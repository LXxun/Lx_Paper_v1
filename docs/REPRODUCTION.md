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

The guarded final-test and recurrent follow-up runners additionally require checkpoints, manifests and locks. The historical execution locks are not in this code-only release. Do not fabricate or bypass them: audit the exact assets and implementation for your own run and record new hashes. Public portability changes alter code hashes from the historical execution. `docs/SOURCE_PROVENANCE.json` records the fetched source hashes and public export hashes.

## Table and figure scripts

With separately supplied numerical result archives, table templates and displayed table references, these entries reconstruct archived numbers:

```sh
python source/reproduce_main_tables.py --out outputs/main_tables --check
python source/reproduce_recurrent_tables.py --out outputs/recurrent_tables --check
```

They expect the original `data/`, `tables/` and `reproduction/INPUT_HASHES.json` layout described in their source. Those result artifacts are intentionally absent from this release. Confidence intervals are copied from archived summaries, not re-estimated by these display scripts. `source/build_heldout_matrix.py` likewise requires the archived result inputs and matplotlib.

Before publication, the scripts were tested against the local archives (eight tables / 36 rows). The public code-only checkout cannot independently repeat those checks without the omitted assets. Original model evaluation and raw-record bootstrap implementations are provided for inspection and use with independently provisioned inputs.

## Interpretation

The original six contrasts and four recurrent follow-up contrasts are separate families. The latter reused the original test cohort after inspecting the original results, shares the tokenizer/evidence head and changes pretraining. Its implementation clarification and numerical amendment document bounded-output selection and the uniform FP32 restart; no claim of pristine prospective preregistration or attribution-method superiority is made.
