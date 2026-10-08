# Minimal numerical audit bundle

This bundle contains author-generated numerical results and anonymous video-cluster sufficient statistics for **Evaluating Multimodal Sentiment Explanations across Input and Representation Spaces**, by Xun Liu, Aodi Liu and Weibing Wan.

## One command

Requires Python 3.9 or later and NumPy 1.22 or later. No GPU, model weights, PyTorch, dataset download or network access is needed. NumPy 1.26.4 was used in the historical analyses; the audit report records the installed version.

```sh
python audit_materials/reproduce.py --out reproduced_audit
```

Use a new output directory outside this bundle. All released inputs and scripts are checked against `MANIFEST.json` before execution. The command produces:

- Eight LaTeX tables (36 numerical rows), checked against their manuscript row content, and the two validation hybrid absolute-gap means.
- Fourteen bootstrap intervals recalculated from video sums and counts, checked against archived full-precision estimates and interval endpoints (absolute tolerance 1e-12).
- `AUDIT_REPORT.json` with results, effective sample and video counts, resampling seeds, number of draws, weighting and software versions.

The three inferential families remain separate:

| Family | Cohort and averaging | Bootstrap | Adjusted interval |
|---|---|---|---|
| Original six | Final test; 3 heads, 3 seeds and 2 substitutions averaged per segment; segment weighting | 50,000 whole-video draws; MOSI seed 2026100201, MOSEI 2026100202 | 99.1667%, six-comparison Bonferroni |
| Recurrent four | Post-original-test follow-up on the same cohort; 3 seeds and 2 substitutions averaged per segment; segment weighting | 50,000 whole-video draws; MOSI seed 2026100701, MOSEI 2026100702 | 98.75%, four-comparison Bonferroni |
| MulT four | Subsequent validation-only follow-up; 3 seeds and 2 substitutions per segment, then equal video weighting | 20,000 whole-video draws; seed 20261005 for each contrast | 98.75%, four-comparison Bonferroni |

MOSI has 686 segments and 31 videos in the original/recurrent cohorts, and 229 segments and 10 videos in MulT validation. MOSEI has 300 segments and 300 videos in each stated text cohort. Families are not pooled. The MulT signed-retention metric remains distinct from the original and recurrent negative absolute reconstruction loss.

## Anonymous sufficient statistics

`clusters/{original,recurrent,mult}.json` contains one row per video cluster: a local ordinal group code, segment count `n`, and sums of already averaged segment contrasts. Group codes have meaning only within a dataset and family. They do not identify source videos, speakers, sample IDs, timestamps or locations; no lookup mapping is released. Their order preserves the archived cluster order so the deterministic bootstrap draws remain reproducible.

For segment-weighted analyses, a sampled set of video clusters contributes the sum of its contrast sums divided by the sum of its segment counts. MulT first divides each cluster sum by its count and then averages sampled video means. Thus full segment records are unnecessary for recalculating these particular estimates and intervals. Video grouping is the recorded statistical unit, not an assertion of speaker-level independence or a formal privacy guarantee.

Original columns are deletion matching, retention matching, reference-minus-target selection-change indicator, reference-change indicator, and target-change indicator. The last three use proportions, not percentage points; the display script multiplies the relevant contrast by 100. Recurrent and MulT columns are deletion and retention matching. All calculations use the declared original signs and metrics. Random-seed/head variability is not newly resampled.

## Table inputs and scope

`data/` retains only numerical fields used for reconstruction: split and analysis counts, aggregate prediction metrics, configuration means, primary summaries, validation hybrid means and recurrent validation summaries. Sensitive and unnecessary execution metadata were removed; new hashes describe this publication export, not altered historical locks. `source/` and `tables/` reconstruct and verify the eight existing numeric tables; the new literature-positioning table is qualitative and is not counted among them. LaTeX assigns table numbers dynamically.

The display scripts copy archived confidence intervals; `reproduce.py` separately recomputes the fourteen intervals from the anonymous cluster inputs. Agreement tests the aggregation and inference arithmetic. This bundle cannot recover individual rankings, check feature extraction, verify original model outputs, or rerun training. Raw input data, labels, participant records, feature tensors, trained and pretrained weights, and source-ID mappings are not distributed. Full model reproduction still requires separately permitted assets and the experiment code. This package does not turn the post-original-test follow-ups into independent replications.

See `SHARING_REVIEW.md` for the scope of the redistribution review and `PROVENANCE.json` for archive and extraction fingerprints.
