# Controlled Probing Protocol

The probing pipeline evaluates the validation split with three complementary
intervention families. The transformations measure sensitivity to the
implemented changes; they are not treated as selective causal interventions.

## Bundle A: Cue Removal

- `text_masked`: EasyOCR text regions followed by Telea inpainting.
- `vehicle_masked`: YOLOv8n car, motorcycle, bus, and truck boxes followed by
  Telea inpainting.
- `sky_masked` and `vegetation_masked`: SegFormer-B0 ADE20K masks followed by
  Telea inpainting.
- `{cue}_random_control`: a deterministic compact random mask containing
  exactly the same number of pixels as that image's corresponding cue mask.

## Bundle B: Appearance Reduction

- `edges`: three-channel Canny edge maps.
- `macro_blur`: Gaussian blur with sigma 15.

Edges and blur retain multiple kinds of information and are only coarse
structure proxies.

## Bundle C: Layout Disruption

- `patch_shuffled_16`
- `patch_shuffled_32`
- `patch_shuffled_64`

Images are resized to 224 by 224 before deterministic patch permutation. Border
pixels outside a complete patch grid are preserved for every patch size. The
Caltech101 control calls the same implementation.

## Metrics

Prediction switch rate is the sample-level label change rate:

```text
SR = mean(prediction_clean != prediction_intervention) * 100
```

Retention is the transformed-to-clean accuracy ratio:

```text
Retention = Accuracy_intervention / Accuracy_clean
```

The result files also contain transformed accuracy and percentage-point
accuracy drop. These quantities are separate from SR.
