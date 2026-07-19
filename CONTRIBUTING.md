# Contributing to EasyStudy

Thanks for your interest in improving EasyStudy! This project welcomes contributions from
researchers, students, and practitioners.

## Ways to contribute
- **New plugins** (study templates), **data loaders**, **algorithms**, **preference-elicitation
  methods**, or **evaluation metrics** — see the "Development" section of the [README](README.md).
- **Bug fixes and core improvements.**
- **Docs and tutorials.**

## Development setup
```bash
git clone https://github.com/pdokoupil/EasyStudy.git
cd EasyStudy
uv sync --extra dev                # lightweight core + dev tools (creates .venv from uv.lock)
# optional heavy backends:
# uv sync --extra recbole          # RecBole model zoo (BPR, LightGCN, NGCF, NeuMF, DMF)
# uv sync --extra tensorflow       # VAE / RBM / TF-Recommenders algorithms
# uv sync --extra lenskit          # LensKit wrappers
python scripts/fetch_data.py --dataset ml-latest-small   # small, fast demo dataset
cd server && uv run flask --debug run
```
No [uv](https://docs.astral.sh/uv/)? `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` works too.
See the [README](README.md) for the Docker / `docker compose up` path.

## Branch model
- `main` is the canonical, most up-to-date branch — target your PRs here unless told otherwise.
- `ndbi021` is the actively-maintained lightweight branch used for teaching (fewer dependencies);
  core fixes generally land on both.
- Other study-specific branches (`feature/journal`, `feature/grs2024`, …) hold the plugins used
  for individual papers and are kept for reference only.

## Pull request checklist
- [ ] `bash scripts/check_diff.sh` passes locally — formats/lints **only the Python files your
  branch changes** (vs. `origin/main`) with black + ruff, then runs the full test suite. This is
  exactly what the CI `lint-diff` job re-checks on your PR (scoped the same way, so it won't
  fail you for pre-existing style debt elsewhere in the repo). Use `--fix` to auto-format;
  `--no-tests` to skip the test run for a fast lint-only loop.
- [ ] Tests pass, and new behavior has a test.
- [ ] If you changed models, you added a migration (`flask db migrate`).
- [ ] Docs/README updated if you changed user-facing behavior.
- [ ] New heavy dependencies go behind an extra in `pyproject.toml`, not into core.
- [ ] If you changed dependencies, `uv.lock` is regenerated (`uv lock`) and committed — CI fails
  otherwise (`uv lock --check`).

## Reporting bugs
Open an issue using the bug template. Include your OS, Python version, install extras, and steps to
reproduce.
