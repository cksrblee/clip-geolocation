# Experiment Definitions

## Adaptation Strategies

All runs use OpenCLIP ViT-L-14 with OpenAI weights and their QuickGELU
activation, seed 42, batch size 8, AdamW with weight decay 0.01, 10 epochs,
cosine annealing, cross-entropy loss, automatic mixed
precision, and the checkpoint with the highest validation accuracy.

| Strategy | Trainable components | Learning rate |
| --- | --- | ---: |
| Zero-shot | none | - |
| LP-T | text-score feature scale and bias | `1e-4` |
| LP-C | linear classifier | `1e-4` |
| PU | final visual block, `ln_post`, classifier | `1e-5` |
| LoRA | rank-8 attention/MLP adapters, classifier | `1e-4` |
| Full-FT | complete visual encoder, classifier | `1e-5` |

The classifier strategies operate on normalized CLIP image embeddings. Text
features remain fixed and are used directly only by Zero-shot and LP-T.

## Evaluation Sets

- Adaptation accuracy and centroid error: test split.
- Intervention probing and cue ANOVA: validation split.
- Prompt controls: test split for all four prompt variants.
- Caltech101: 2,000 seed-42 samples and 102 zero-shot candidate labels,
  including the background distractor.

Centroid error is computed from each image coordinate to the centroid of the
predicted region. Prompt descriptors are fixed in
`configs/prompt_controls.json`; the loader rejects descriptors containing a
target region name.

Running the scripts regenerates the table inputs under these definitions. The
repository does not use contrastive training in the reported experiment set.
