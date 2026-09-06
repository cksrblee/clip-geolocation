# Dataset Specification

## Scope

The benchmark contains 9,085 Google Street View perspective images from eight
Greater Los Angeles regions:

| Region ID | Region | Sampled images |
| --- | --- | ---: |
| `downtown_la` | Downtown Los Angeles | 998 |
| `hollywood` | Hollywood | 1,197 |
| `beverly_hills` | Beverly Hills | 1,746 |
| `santa_monica` | Santa Monica | 1,248 |
| `venice_beach` | Venice Beach | 599 |
| `koreatown` | Koreatown | 599 |
| `pasadena` | Pasadena | 1,496 |
| `long_beach` | Long Beach | 1,202 |

The region bounding boxes and class order are defined in
`configs/la_county_regions.json`.

## Sampling and Splits

The raw metadata contains 28,550 eligible images. Sampling uses the largest
image density supported by every region, 194.49 images/km2, then applies a
seed-42 region-stratified 70/15/15 image-level split:

- train: 6,359
- validation: 1,363
- test: 1,363

`scripts/00_build_splits.py` writes the sampled metadata, sampling summary,
three splits, and the test-to-train spatial-overlap audit.

## Spatial Overlap

The split is not geographically disjoint. In the checked-in split, 755 test
images share an exact requested coordinate with training, 1,086 are within 50 m
of a training coordinate, and 277 are farther than 50 m. The evaluation
therefore measures viewpoint variation near known locations, not generalization
to unseen geographic areas.

Each record retains filename, latitude, longitude, heading, and region ID. Raw
Street View images are not included in the repository.
