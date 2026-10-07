# Lx_Paper_v1

Code for **Evaluating Multimodal Sentiment Explanations across Input and Representation Spaces**, by Xun Liu and Weibing Wan.

This code-only release contains the study's model, training, attribution, intervention-evaluation and analysis implementations. It investigates how comparisons change between input re-encoding and frozen evidence; it does not claim a universally superior attribution algorithm.

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

## Reproduction scope

This is a **code-only** publication. MOSI/MOSEI feature packages, pretrained and trained weights, participant-level records, aggregate numerical result archives, private execution logs and manuscripts are not included.

Full model experiments require separately obtained data and pretrained assets, appropriate external dependencies, and newly audited execution manifests. The reconstruction scripts require the author's result archives; they are not runnable end-to-end from this checkout alone. See [REPRODUCTION.md](docs/REPRODUCTION.md) for exact requirements. Public code improves inspectability but does not make the complete experiment independently reproducible without those assets.

## Dependencies and attribution

The research environment used Python 3.10, PyTorch 2.6.0/CUDA 12.4 and NumPy 1.26.4; see `requirements-models.txt`. Third-party dependencies retain their own licenses and are not vendored. No additional redistribution license is granted by this initial code-availability release.

- MMSA: https://github.com/thuiar/MMSA
- CMU Multimodal SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- Transformers: https://github.com/huggingface/transformers
