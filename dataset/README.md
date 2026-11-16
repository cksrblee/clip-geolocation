# Dataset Directory

Checked-in CSV files define the 9,085-image, eight-region experiment dataset.
The image files themselves remain local.

```text
metadata.csv                    eligible raw metadata
sampled_metadata.csv            area-normalized sample
sampling_summary.csv            area, availability, and target density
train_metadata.csv              6,359 training records
val_metadata.csv                1,363 intervention/validation records
test_metadata.csv               1,363 final evaluation records
test_spatial_overlap.csv         per-test nearest-train distance
spatial_overlap_summary.csv      aggregate coordinate-overlap counts
images/                          local Street View JPEGs (gitignored)
```

The core columns are `filename`, `region_id`, `latitude`, `longitude`, and
`heading`. Run `python3 scripts/00_build_splits.py` to regenerate all sampled,
split, and overlap files from `metadata.csv`.
