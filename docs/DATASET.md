# Dataset Specification

## Scope

The benchmark contains **9,085** Google Street View perspective images across eight Greater Los Angeles regions (four incorporated cities and four recognized neighborhoods/districts):

| Region ID | Region | Area ($\text{km}^2$) | Eligible raw | Sampled ($N$) | Same-GPS Test Overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| `downtown_la` | Downtown Los Angeles | 5.13 | 4,000 | 998 | 41.6% (62/149) |
| `hollywood` | Hollywood | 6.16 | 4,000 | 1,197 | 52.2% (94/180) |
| `beverly_hills` | Beverly Hills | 8.98 | 4,000 | 1,746 | 69.1% (181/262) |
| `santa_monica` | Santa Monica | 6.42 | 4,000 | 1,248 | 47.6% (89/187) |
| `venice_beach` | Venice Beach | 3.08 | 3,921 | 599 | 28.9% (26/90) |
| `koreatown` | Koreatown | 3.08 | 599 | 599 | 85.6% (77/90) |
| `pasadena` | Pasadena | 7.69 | 4,000 | 1,496 | 63.6% (143/225) |
| `long_beach` | Long Beach | 6.18 | 4,000 | 1,202 | 46.1% (83/180) |

Region bounding boxes and class order are defined in `configs/la_county_regions.json`.

## Sampling and Splits

Of the 28,550 raw images in `dataset/metadata.csv`, 30 lie outside the eight target regions, leaving **28,520** eligible images. Sampling uses the largest spatial density supported by every region, $\rho^\star = \min_k(n_k / A_k) = 194.49\text{ images/km}^2$ (determined by Koreatown), followed by a seed-42 region-stratified 70/15/15 image-level split:

- **Train**: 6,359 images
- **Validation**: 1,363 images
- **Test**: 1,363 images

`scripts/00_build_splits.py` regenerates the sampled metadata, sampling summary, three split CSVs, and the test-to-train spatial-overlap audit.

## Spatial Overlap

Because perspective views were collected at four orthogonal headings ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) per location without coordinate-grouped splitting, the image-level split is geographically overlapping:

- **755** test images (**55.39%**) share an exact requested coordinate with training;
- **1,086** test images (**79.68%**) lie within $50\text{ m}$ of a training coordinate;
- **277** test images (**20.32%**) are separated by more than $50\text{ m}$.

The benchmark therefore evaluates viewpoint variation near known locations rather than geographically disjoint generalization. Raw Street View JPEGs are not redistributed.
