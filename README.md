# SMFS Catalog

A desktop application for cataloguing and analysing single-molecule force
spectroscopy (SMFS) AFM force curves.

It registers Igor Binary Wave (`.ibw`) force curves into a searchable SQLite
catalog, decides which curves contain rupture events, locates the ruptures,
fits worm-like-chain (WLC) models to them, and provides interactive tools for
exploring the resulting population — 2-D histograms, distribution fitting,
PCA and clustering — over whatever cohort you select.

It is built for the working pattern of a lab: point it at a directory of
curves, let it work through them, then spend your time on the curves and the
statistics rather than on bookkeeping.

## What it does

- **Catalogues curves.** Registers `.ibw` files with their metadata, tracks
  which are readable, which failed qualification, and which have been analysed.
- **Classifies automatically.** Baseline correction, deflection-sensitivity
  handling and landmark detection decide event from non-event; results are
  cached against the exact parameter set and code version that produced them.
- **Finds and fits ruptures.** Detection regions within each curve, ruptures
  within those, and a WLC fit per segment — with contour length, persistence
  length and honest uncertainty rather than a single unqualified number.
- **Explores the population.** 2-D histograms, distribution fitting with
  bootstrapped confidence intervals, Gaussian mixture models, PCA, and
  per-class overlays.
- **Exports reproducibly.** Every export carries a manifest recording the
  contributing files, the settings in force, the app version and the code
  version — so a figure can be traced back to what produced it.

## Installing

Requires [conda](https://docs.conda.io/) (Miniconda is enough). The
environment file is the single dependency list; there is no `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate smfs-catalog
python run_dashboard.py
```

Pass a path to use a specific catalog database:

```bash
python run_dashboard.py /path/to/smfs_catalog.db
```

With no argument it uses a per-user default location, which `$SMFS_DB_PATH`
overrides.

### Prebuilt applications

Standalone builds for Windows, macOS and Linux are attached to each
[release](../../releases) — no Python or conda needed. They are unsigned, so
Windows shows a SmartScreen prompt and macOS requires right-click → Open on
first launch. The release notes give the details per platform.

## Documentation

- [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) — how to use the application
- [`docs/SPEC_what_the_app_does.md`](docs/SPEC_what_the_app_does.md) — what
  the analysis pipeline does, stage by stage
- [`docs/CODEBASE_ARCHITECTURE.md`](docs/CODEBASE_ARCHITECTURE.md) — module map
- [`docs/UNCERTAINTY.md`](docs/UNCERTAINTY.md) — how uncertainties are computed
  and what they do and do not mean

## Tests

```bash
conda activate smfs-catalog
python -m pytest -q
```

The suite is the specification: most tests exist because a specific behaviour
was got wrong once, and each explains in its docstring what it is protecting.

## Built with

Python, PyQt6, pyqtgraph, NumPy, SciPy, scikit-learn and
[igor2](https://pypi.org/project/igor2/) for reading Igor waves. SQLite for
the catalog.

## License

Copyright (C) 2026 Joseph Hamill

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See [`LICENSE`](LICENSE) for the full text.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.
