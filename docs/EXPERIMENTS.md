# Experiment Definitions

## Adaptation Strategies

All runs use OpenCLIP `ViT-L-14` (`openai` weights) with its default $224 \times 224$ resize, center crop, and normalization, seed 42, batch size 8, AdamW with weight decay `0.01`, 10 epochs, cosine annealing ($T_{\max}=10$), unweighted cross-entropy loss, automatic mixed precision, and the checkpoint achieving the highest validation accuracy.

| Strategy | Trainable components | Scoring mechanism | Learning rate |
| --- | --- | --- | ---: |
| `Zero-shot` | none | Frozen text similarity | — |
| `LP-T` | Image-feature scale and bias | Frozen text similarity | `1e-4` |
| `LP-C` | Linear classifier ($D \to 8$) | Linear head | `1e-4` |
| `PU` | Final ViT block (`resblocks[-1]`) & `ln_post` | Frozen text similarity | `1e-5` |
| `LoRA` | PEFT rank-8 adapters ($\alpha=16$, dropout `0.1`) on `q/k/v/out_proj`, `c_fc`, `c_proj` | Frozen text similarity | `1e-4` |
| `Full-FT` | Complete visual encoder | Frozen text similarity | `1e-5` |

Text features are computed from `"a Google Street View photo in {Region}, Los Angeles"` and remain frozen across all strategies.

## Evaluation Sets and Metrics

- **Original-image adaptation performance (Table I, Fig. 3a)**: evaluated on the held-out `test` split (`N = 1,363`). Centroid error (`CE`) measures the mean geodesic distance (km) between predicted and ground-truth region centroids (`0 km` when predicted correctly).
- **Intervention probing & visual-cue ANOVA (Tables I–II, Fig. 3b–d, Sec. V-A)**: evaluated on the `val` split (`N = 1,363`).
- **Zero-shot prompt controls (Table I)**: `P0_original`, `P_len` (length-matched filler), `P_visual_only` (region-name-free descriptors from `configs/prompt_controls.json`), and `P_swap` (cyclically reassigned descriptors).
- **Non-geographic control (Table III, Fig. 3d)**: 2,000 seed-42 Caltech-101 samples resized to $224 \times 224$ and evaluated zero-shot under Clean, Patch-16, Patch-32, and Patch-64 conditions.
