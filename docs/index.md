# EasyStudy

**The fastest way to build, run, and analyze interactive user studies — for recommender systems and
any other item-based interactive system.**

EasyStudy is a modular, highly extensible framework that ships out-of-the-box functionality for every
phase of a study: data loading & presentation, preference elicitation, baseline algorithms,
questionnaires, and results comparison. A simple study can be deployed **in a few clicks from a web UI**,
while complex designs stay fully programmable via **plugins**.

> Credo: *"Make simple things fast and hard things possible."*

## Why EasyStudy?

- **Batteries included.** Unlike tools that make you bring your own recommender, EasyStudy ships
  datasets, algorithms (e.g. EASE, popularity, VAE, LensKit wrappers), and preference-elicitation
  methods — so you can run a *personalized* study with zero external services.
- **Domain-generic.** Built for recommender systems, already used for **data-visualization** studies and
  **university teaching**. If your study shows people *items*, records how they *interact*, and asks them
  *questions*, EasyStudy fits.
- **Extensible by design.** Add datasets, algorithms, elicitation methods, metrics, or whole study flows
  by subclassing a base class or dropping in a plugin.
- **Lightweight core.** `pip install easystudy` pulls a small pure-Python stack; heavy backends
  (TensorFlow, LensKit) are opt-in extras.

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart](quickstart.md)** — a running study in 15 minutes.
- :material-lightbulb: **[Concepts](concepts.md)** — plugins vs studies vs instances.
- :material-cursor-default-click: **[Create a study (no code)](create-a-study.md)** — from the admin UI.
- :material-puzzle: **[Extending fastcompare](extending.md)** — add a dataset / algorithm / metric.
- :material-package-variant: **[Write your first plugin](first-plugin.md)**.
- :material-database: **[Data model & exporting results](data-model.md)**.

</div>

## Citation

If you use EasyStudy in your research, please cite the [RecSys'23 paper](https://doi.org/10.1145/3604915.3610640)
(see `CITATION.cff`).
