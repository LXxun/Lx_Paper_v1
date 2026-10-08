# Lx_Paper_v1

Code for **Evaluating Multimodal Sentiment Explanations across Input and Representation Spaces**, by Xun Liu, Aodi Liu and Weibing Wan. Xun Liu and Aodi Liu share first authorship.

This release contains the study's model, training, attribution, intervention-evaluation and analysis implementations. It investigates how comparisons change between input re-encoding and frozen evidence; it does not claim a universally superior attribution algorithm.

## Contents

- `experiments/benchmarks/source/tei_kan/`: model, KAN components, losses, attribution, eligible-unit selection and training utilities.
- `experiments/benchmarks/compact_stage/`: deterministic evidence heads and compact explanation code.
- `experiments/benchmarks/final_test_20261002/`: original guarded evaluation and analysis sources.
- `experiments/independent_encoder_20261007/`: BiLSTM follow-up training/evaluation/analysis and protocol amendments.
- `experiments/mult_extension/`: adapted-MulT interface and validation comparison.
- `source/`: table reconstruction and held-out matrix plotting code.
- `tests/`: CPU synthetic checks, with no dataset or checkpoint required.

## Quick check

After installing compatible NumPy and PyTorch:

```sh
python tests/test_core.py
```

The two checks cover WordPiece grouping/continuous-window rules and model evidence-to-output consistency on synthetic inputs. They passed in the original Linux research environment. Python syntax checks also passed; a full GPU retraining/evaluation of this portable export has not been performed.

## Numerical audit: no datasets or weights required

```sh
python audit_materials/reproduce.py --out outputs/numerical_audit
```

Requires Python and NumPy. Use a new output directory. The public audit bundle reconstructs **8 tables / 36 numerical rows** and recomputes **14 adjusted bootstrap intervals** from anonymous per-video contrast sums and segment counts. It preserves the original six-contrast family and the two separate four-contrast follow-up families. See [audit instructions](audit_materials/README.md), [sharing review](audit_materials/SHARING_REVIEW.md), and the [verified report](audit_materials/VERIFIED_REPORT.json).

This audit checks numerical aggregation and inference arithmetic, not training or original model inference. MOSI/MOSEI media, transcripts, labels, feature tensors, original identifiers, individual intervention records and weights are excluded. Full model experiments still require separately obtained assets and newly audited execution manifests. See [REPRODUCTION.md](docs/REPRODUCTION.md).

## Dependencies and attribution

The research environment used Python 3.10, PyTorch 2.6.0/CUDA 12.4 and NumPy 1.26.4; see `requirements-models.txt`. Third-party dependencies retain their own licenses and are not vendored. No license over upstream datasets or third-party assets is granted by this release.

- MMSA: https://github.com/thuiar/MMSA
- CMU Multimodal SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- Transformers: https://github.com/huggingface/transformers
