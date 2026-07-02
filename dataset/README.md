# dataset

## Purpose

This directory stores dataset files, dataset notes, and local data conventions for this project.

The current main line is a small `Stable Diffusion`-style pipeline, so dataset selection is driven by:

- anime face generation
- a `VAE -> latent diffusion` workflow
- keeping relatively high-quality face / head samples

## Current Dataset Choice

The current preferred route is `DAF:re / DAFB`.

Why this choice:

- it is a curated anime face / head dataset rather than an arbitrary small archive
- it is tied to the `Danbooru` family of datasets, so quality and consistency are usually better than mixed-source collections
- it is large enough for later `VAE` and latent diffusion experiments

## Confirmed Facts

- dataset name: `DAFB`
- upstream background: related to `DAF:re (DanbooruAnimeFaces:revamped)`
- image resolution: `128x128`
- scale:
  - the final `DAF:re Faces` release in the paper has about `463,437` images
  - the currently exposed package name on `Hugging Face` is `daf.tar.gz`
- the `Hugging Face` dataset page shows roughly `13.34 GB`

## Current Download Strategy

Instead of cleaning from full `Danbooru2021`, this project currently prefers downloading an already curated package.

Current local data layout:

- `raw/fullMin256/`
- `raw/train.csv`
- `raw/train_val.csv`
- `raw/classid_classname.csv`

Why:

- download cost is more controllable
- we can move into the `VAE` and `Stable Diffusion` workflow faster
- we avoid spending early project time on upstream large-scale cleaning

## Directory Layout

```text
dataset/
  README.md
  README_CN.md
  raw/
    README.md
    classid_classname.csv
    train.csv
    train_val.csv
    fullMin256/
```

Notes:

- `raw/` currently stores the extracted DAF data directly
- `raw/README.md` is the upstream dataset note and remains in the repo; images, CSV files, and other larger assets are still treated as local data files
- `raw/fullMin256/` is the default image directory used by the VAE
- training code scans images recursively under `raw/fullMin256/` and does not rely on a single flat directory

## Relation To Training Resolution

The current main line fixes the VAE input resolution at `128x128`.

Recommended setup:

- keep original data under `dataset/raw/fullMin256/`
- resize to `128x128` during VAE training
- keep later latent shape fixed at `[batch, 4, 16, 16]`

## Future Notes

More details will be added here later:

- extraction steps
- train / validation split strategy
- whether more abnormal-sample filtering is needed
