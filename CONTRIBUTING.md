# Contributing to EasyStudy

Thanks for your interest in improving EasyStudy! This project welcomes contributions from
researchers, students, and practitioners.

## Ways to contribute
- **New plugins** (study templates), **data loaders**, **algorithms**, **preference-elicitation
  methods**, or **evaluation metrics** — see the "Development" section of the [README](README.md).
- **Bug fixes and core improvements.**
- **Docs and tutorials.**
- **Student projects** — see [`04_STUDENT_CONTRIBUTIONS.md`](https://github.com/pdokoupil/EasyStudy)
  (planning docs) for well-scoped tasks.

## Development setup
```bash
git clone https://github.com/pdokoupil/EasyStudy.git
cd EasyStudy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # lightweight core + dev tools
# optional heavy backends:
# pip install -e ".[tensorflow]"   # VAE / TFRS algorithms
# pip install -e ".[lenskit]"      # LensKit wrappers
python scripts/fetch_data.py --dataset all   # download datasets + images
cd server && flask --debug run
```
See the [README](README.md) for the Docker / `docker compose up` path.

## Branch model
- `main` is the canonical, most up-to-date branch — target your PRs here unless told otherwise.
- Study-specific branches (`feature/journal`, `feature/grs2024`, `ndbi021`, …) hold the plugins
  used for individual papers/courses; they are kept for reference. **Core fixes belong on `main`.**

## Pull request checklist
- [ ] Code is formatted (`black .`) and lint-clean (`ruff check .`).
- [ ] Tests pass (`cd server && pytest`), and new behavior has a test.
- [ ] If you changed models, you added a migration (`flask db migrate`).
- [ ] Docs/README updated if you changed user-facing behavior.
- [ ] New heavy dependencies go behind an extra in `pyproject.toml`, not into core.

## Reporting bugs
Open an issue using the bug template. Include your OS, Python version, install extras, and steps to
reproduce.
