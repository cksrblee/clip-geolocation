# Controlled Probing Protocol

The probing pipeline evaluates the validation split (`N = 1,363`) across three complementary intervention families. Each transformation alters multiple visual properties, so results are interpreted jointly as sensitivity to the implemented perturbations rather than isolated causal attributions.

## Bundle A: Semantic & Environmental Cue Removal

- `text_masked`: EasyOCR (CRAFT) text polygons followed by Telea inpainting.
- `vehicle_masked`: YOLOv8n MS-COCO vehicle boxes (`car`, `motorcycle`, `bus`, `truck`) followed by Telea inpainting.
- `sky_masked` and `vegetation_masked`: SegFormer-B0 ADE20K masks followed by Telea inpainting.
- `random_rectangle`, `sky_random_control`, `vegetation_random_control`: area-matched square random-mask controls inpainted with the same Telea algorithm to separate semantic cue removal from generic area occlusion.

One-way ANOVA across the eight regions confirms significant regional variation in mask coverage for all four cues ($p < 0.01$), led by vegetation ($F = 35.04$) and sky ($F = 10.93$).

## Bundle B: Appearance Reduction (Structure Proxies)

- `edges`: three-channel Canny edge maps (`100`, `200` thresholds).
- `macro_blur`: Gaussian macro-blur ($\sigma = 15$).

Edges and blur reduce fine appearance detail while retaining coarse spatial organization, outlines, and (for blur) global color palettes.

## Bundle C: Spatial-Layout Disruption

- `patch_shuffled_16`, `patch_shuffled_32`, `patch_shuffled_64`: deterministic $n \times n$-pixel grid permutation on $224 \times 224$ images.
- `patch_mixed`: cross-region $32 \times 32$ checkerboard interleaving.

## Metrics

For Bundle A and Bundle C, `SR` (Shortcut Reliance / Intervention-induced Accuracy Drop) measures the percentage-point change in accuracy:

```text
SR = Accuracy_original - Accuracy_intervention
```

Sample-level prediction instability (`mean(pred_original != pred_intervention) * 100`) is additionally logged as `Prediction_Switch_Rate_pct`.

For Bundle B, structural-sufficiency retention (`SS` / `Retention`) normalizes transformed accuracy by each model's clean accuracy:

```text
Retention (SS) = Accuracy_intervention / Accuracy_original
```
