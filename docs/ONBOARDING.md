# Onboarding for the LaboGraph teams

SMFS Catalog is the desktop application the LaboGraph projects start from.
This page explains how to get the code and what its terms mean.

## Getting the code

The repository is public. Use it as the reference for what the application
does; build your own application in your own repository.

Work from the tag `labograph-baseline`. A tag names one exact commit, so the
code you read stays the same while development here continues.

### Copy it to GitLab

1. In GitLab: **New project** → **Import project** → **Repository by URL**.
2. Paste `https://github.com/josephhamill/smfs_catalog.git`.

This copies the code, its history and its tags, not the issues or pull
requests.

### Run it

You need [conda](https://docs.conda.io/) (Miniconda is enough). Clone your
GitLab copy, with its address in place of the one below:

```bash
git clone <your GitLab copy>
cd smfs_catalog
git switch --detach labograph-baseline
conda env create -f environment.yml
conda activate smfs-catalog
python run_dashboard.py
python -m pytest -q
```

The first launch creates an empty catalog database. The
[walkthrough video](https://github.com/josephhamill/smfs_catalog/releases/download/v1.3.0/smfs_catalog_walkthrough_v1-3-0.mp4)
(~30 min) shows the application in use. It was recorded on v1.3.0, so some
windows have changed since.

### Propose a change

Bring questions and proposals to the project meetings first. Once a change is
agreed, fork this repository on GitHub and open a pull request from your fork.

## Glossary

### The data

- **Force curve** — one recorded measurement: the cantilever's deflection as
  the probe approaches the surface and retracts from it. Each curve is one
  Igor Binary Wave (`.ibw`) file.
- **Catalog** — the SQLite database. It holds each file's path, metadata and
  analysis results. The curve samples stay in the `.ibw` files.
- **Experimentalist** — the person who recorded a curve.
- **Scope** — the set of files the dashboard filters currently select.
- **Queue** — the files sent for analysis.

### The analysis of one curve

- **Analysis parameters** — the settings the analysis runs with. Results are
  stored against the exact parameter set and code version that produced them.
- **Profile** — one experimentalist's analysis parameters. Each
  experimentalist's curves are analysed under their own profile.
- **invOLS** — inverse optical lever sensitivity: the factor that converts the
  detector signal to cantilever deflection in nanometres.
- **Landmarks** — reference points found on each curve, such as contact with
  the surface, snap-off from it, and the onset of loading.
- **Event / non-event** — the automatic verdict on a curve. An event curve
  contains at least one rupture. A curve can also be *unavailable* or
  *unusable*.
- **ROI** (region of interest) — one excursion of the force away from the
  baseline and back.
- **Rupture** — a sudden drop in force inside an ROI, where a bond broke or
  the molecule detached.
- **Segment** — the stretch of curve between two consecutive landmarks in an
  ROI: onset to first rupture, first rupture to second, and so on. Each
  segment gets one chain-model fit.
- **Ultimate / penultimate segment** — which segment's values are reported
  for a curve: the last one or the one before it. A manual **Primary /
  Secondary** pick overrides this for one curve.
- **Chain model** — the polymer-elasticity model fitted to a segment:
  worm-like chain (WLC, in Marko-Siggia, Bouchiat and extensible forms) or
  freely jointed chain (FJC, plain and extensible). The fits give a contour
  length `l_c` and a persistence length `l_p` or Kuhn length `b`.

### The analysis of many curves

- **Variable** — a named per-curve quantity, such as rupture force or contour
  length. All of them are declared in `smfs_catalog/variables.py`.
- **Criterion** — a lower or upper bound on a variable, set per
  experimentalist.
- **Hit / non-hit** — a hit is an event curve that passes every criterion.
  The verdict is never stored; it is derived from the current bounds each
  time.
- **Population** — the hits or the non-hits of the current scope: the set of
  curves the statistics windows work on.
- **2DH** — a two-dimensional histogram of force against extension, summed
  over a population. The *physical* 2DH is in nanometres and piconewtons,
  aligned at a chosen landmark. The *normalized* 2DH is in the dimensionless
  coordinates of the WLC model.
- **Drop reason** — why a curve is missing from a view, for example because
  its fit failed. Every dropped curve carries one.
- **Manifest** — the file written with every export. It records the
  contributing files, the settings, the criteria and the code version.
