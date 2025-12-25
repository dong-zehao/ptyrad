# Repository Guidelines

## Project Structure & Module Organization
- `src/ptyrad/` holds the core Python package (reconstruction pipeline, models, IO, visualization, and utilities).
- `scripts/` contains entry points for running reconstructions (`run_ptyrad.py`) and HPC submission (`slurm_run_ptyrad.sub`).
- `demo/` includes example data, parameter YAMLs, and notebooks (`demo/scripts/`).
- `environment_ptyrad.yml` defines the recommended conda environment; `pyproject.toml` registers the package for editable installs.

## Build, Test, and Development Commands
- `conda env create -f environment_ptyrad.yml` creates the recommended environment (Python 3.10+, PyTorch 2.1+).
- `conda activate ptyrad` activates the environment before any installs or runs.
- `pip install -e .` installs the package in editable mode for local development.
- `python scripts/run_ptyrad.py --params_path "demo/params/tBL_WSe2_reconstruct.yml" --gpuid 0` runs a CLI reconstruction (adjust params and GPU id).
- `jupyter notebook demo/scripts/run_ptyrad_quick_example.ipynb` runs the quick demo in a notebook.

## Coding Style & Naming Conventions
- Follow the existing code style and module layout in `src/ptyrad/`.
- Public functions/classes should include docstrings (per `CONTRIBUTING.md`).
- Use Python snake_case for modules/functions and CapWords for classes (matches current filenames and usage).
- Keep configuration files as `.yml` in `demo/params/` or a similar params directory.

## Testing Guidelines
- No automated test suite is configured in this repository today.
- If you add tests, place them in a new `tests/` folder and document the runner/commands in this file and `README.md`.

## Commit & Pull Request Guidelines
- Development follows a cycle-based workflow: branch from the current `dev-<DATE>` and target PRs back to that branch.
- Branch naming: `feature/*`, `bugfix/*`, `docs/*`.
- PRs are squash-merged into `dev-<DATE>`; hotfixes branch from `main` and are cherry-picked into the dev branch.
- Commit messages have no strict convention in the history; keep them short and descriptive.

## Security & Configuration Tips
- Keep local data paths in YAML params files; do not hardcode absolute paths in source.
- Use GPU selection flags (e.g., `--gpuid`) rather than modifying core modules.
