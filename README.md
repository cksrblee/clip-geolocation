# What Does CLIP Learn for Regional Geolocalization?

Experiment code for *What Does CLIP Learn for Regional Geolocalization? Probing
Visual Cues and Scene Configuration After Adaptation*.

The repository covers the experiments behind Tables I-IX:

- area-normalized sampling and region-stratified 70/15/15 splits;
- ViT-L-14 evaluation with Zero-shot, LP-T, LP-C, PU, LoRA, and Full-FT;
- controlled cue removal, structure proxies, and layout disruption;
- prediction switch rate (SR), accuracy retention (SS), and centroid error;
- regional visual-cue ANOVA, prompt controls, and the Caltech-101 control.

Contrastive training and publication figure generation are not part of the paper
experiment pipeline.

## Layout

```text
configs/
  la_county_regions.json          Region bounding boxes and metadata
  prompt_controls.json            Region-name-free visual descriptors
dataset/
  metadata.csv                    Raw merged metadata (28,550 images)
  sampled_metadata.csv            Area-normalized sample (9,085 images)
  {train,val,test}_metadata.csv   Region-stratified splits
  spatial_overlap_summary.csv     Test-to-train coordinate audit
scripts/
  download_data.py                Optional: download Street View images via Google Maps API
  00_build_splits.py              Area-normalized sampling and 70/15/15 splits (Table I)
  01_train_baselines.py           Train & evaluate 6 adaptation strategies (Tables III-IV)
  01_1_train_lpc.py               Optional LP-C learning rate tuning
  02_build_interventions.py       Generate 13 probing interventions (Bundles A, B, C)
  03_audit_interventions.py       Audit completeness, integrity & mask coverage
  04_evaluate_bundles.py          Evaluate models on 14 conditions (Tables V-VII)
  05_compute_visual_cue_statistics.py  Visual cue coverage & one-way ANOVA (Table VIII)
  06_prompt_grounding.py          Evaluate 4 zero-shot prompt controls (Table IX)
  07_nongeo_control.py            Caltech-101 non-geographic control (Sec. IV-D)
  run_all.sh                      End-to-end experiment pipeline runner
src/
  data/                           Sampling, splits, and image datasets
  models/                         LP-T, LP-C, PU, LoRA, and Full-FT setup
  interventions/                  Masking, structure, and patch transforms
  evaluation/                     Inference, metrics, ANOVA, prompt controls
tests/                            Fast invariant and metric tests
```

Raw images, generated interventions, checkpoints, and results are intentionally
gitignored.

## Setup

```bash
conda env create -f environment.yml
conda activate clip-geo
```

Alternatively:

```bash
python3 -m pip install -r requirements.txt
```

Place the Street View JPEGs in `dataset/images/`. The checked-in metadata uses
the filenames expected by all scripts. (If downloading from scratch, see `python3 scripts/download_data.py --help`).

## Reproduction

Run the full experiment pipeline (no figure generation):

```bash
bash scripts/run_all.sh --model ViT-L-14
```

Use `--skip-splits` to keep existing split CSVs and `--skip-nongeo` to skip the
Caltech-101 download/evaluation.

The individual commands are:

```bash
python3 scripts/00_build_splits.py
python3 scripts/01_train_baselines.py --model ViT-L-14
python3 scripts/01_1_train_lpc.py --model ViT-L-14 --learning-rate 1e-4
python3 scripts/02_build_interventions.py --bundles A B C
python3 scripts/03_audit_interventions.py
python3 scripts/04_evaluate_bundles.py --model ViT-L-14
python3 scripts/05_compute_visual_cue_statistics.py
python3 scripts/06_prompt_grounding.py --model ViT-L-14
python3 scripts/07_nongeo_control.py --model ViT-L-14 --samples 2000
```

`01_1_train_lpc.py` is an optional LP-C-only convergence run; it delegates to
the same trainer and checkpoint format as the main baseline script.

## Controlled Conditions

The evaluation uses the clean image plus 13 interventions (14 conditions total):

| Bundle | Conditions |
| --- | --- |
| A: cue removal | `text_masked`, `vehicle_masked`, `sky_masked`, `vegetation_masked` |
| A: matched controls | `text_random_control`, `vehicle_random_control`, `sky_random_control`, `vegetation_random_control` |
| B: structure proxies | `edges`, `macro_blur` |
| C: layout disruption | `patch_shuffled_16`, `patch_shuffled_32`, `patch_shuffled_64` |

Bundle A uses EasyOCR, YOLOv8n, SegFormer-B0, and Telea inpainting. Random
controls use the measured cue-mask area. `mask_coverage.csv` stores exact mask
pixel counts for auditing and ANOVA.

OpenAI checkpoints are instantiated with their canonical QuickGELU activation.
Supervised runs use batch size 8, AdamW, 10 epochs, cosine annealing, and the
best validation checkpoint. PU, LoRA, and Full-FT use a trainable linear
classifier over normalized image embeddings. LoRA applies rank-8 branches to
the ViT attention and MLP projections.

## Metrics and Outputs

- `results/baseline_results.csv`: test Top-1 and image-to-predicted-centroid error for six strategies.
- `results/baseline_per_region.csv`: per-region test accuracy for all strategies.
- `results/probing_evaluation_results_raw.csv`: accuracy, accuracy drop, SR, and retention for all 14 conditions.
- `results/probing_evaluation_results.csv`: intervention-only Table V-VII metrics.
- `results/visual_cue_*.csv`: image coverage, regional summaries, and ANOVA.
- `results/caltech101_control_results.csv`: clean/scrambled accuracy and relative drop.
- `results/prompt_grounding_results.csv`: four test-split prompt-control accuracies.
- `results/prompt_grounding_prompts.csv`: exact class prompt used in each condition.

SR is the percentage of validation images whose predicted class changes after an
intervention. SS is `Acc(intervention) / Acc(original)`. Accuracy drop is
reported separately as `Acc(original) - Acc(intervention)`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{lee2026what,
  title={What Does CLIP Learn for Regional Geolocalization? Probing Visual Cues and Scene Configuration After Adaptation},
  author={Lee, Changyu and Park, Yeonsoo and Alfarrarjeh, Abdullah and Kim, Seon Ho},
  journal={arXiv preprint arXiv:2608.21761},
  year={2026},
  url={https://arxiv.org/abs/2608.21761}
}
```

## License

MIT. See [LICENSE](LICENSE).
