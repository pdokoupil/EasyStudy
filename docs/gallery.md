# Gallery — studies built with EasyStudy

Real studies deployed with EasyStudy, as evidence that it scales from a class exercise to published
research. (Mirrors how [Informfully](https://informfully.ch) showcases its CH'20/CH'22/DE'22/NL'23
deployments.) See also **[Studies built with EasyStudy](publications.md)** for the full, regularly
updated list of peer-reviewed papers.

<!-- Maintainer note: keep each entry to the schema below — a one-line *what was studied*, the
*venue + year* with a citation/DOI link, the *plugin/branch* used, and ideally one representative
screenshot (either the deployed study UI, or the paper's teaser figure). Put images in
docs/assets/gallery/<slug>.png and reference them with a normal markdown image. -->

## Featured studies

### Rows or Columns? Minimizing presentation bias when comparing multiple RS
- **What:** measured positional/presentation bias across page layouts when comparing several recommenders.
- **Venue:** SIGIR '23 · [DOI:10.1145/3539618.3592056](https://doi.org/10.1145/3539618.3592056)
- **Plugin/branch:** `layoutshuffling` (alternates layouts during the study).

### Controllable multi-objective recommenders (slider UI)
- **What:** compared single- vs multi-objective RS; a slider UI let users set propensity toward
  relevance/novelty/diversity. Yielded the **SM-RS 2.0** user-perceived-qualities dataset.
- **Venue:** ACM TORS · [DOI:10.1145/3754459](https://doi.org/10.1145/3754459) · dataset [osf.io/wsakx](https://osf.io/wsakx)
- **Plugin/branch:** `feature/multiObjectivePlugin` / `slidershuffling`.

### EasyStudy (the framework demo)
- **What:** the original within-user comparison of EASE / MF / kNN baselines on MovieLens & goodbooks.
- **Venue:** RecSys '23 · [DOI:10.1145/3604915.3610640](https://doi.org/10.1145/3604915.3610640)
- **Plugin/branch:** `fastcompare` (`feature/recsys2023`).

### Data-visualization user studies (domain transfer)
- **What:** EasyStudy adapted **outside recommender systems**, for data-visualization experiments —
  concrete proof the framework is domain-generic.
- **Venue:** IEEE — [ieeexplore 10854212](https://ieeexplore.ieee.org/document/10854212) ·
  and Springer chapter [10.1007/978-3-031-49368-3_14](https://doi.org/10.1007/978-3-031-49368-3_14)

### Teaching at Charles University (NDBI021 / NSWI166)
- **What:** used in RS courses both to demo algorithms live and to evaluate students' solutions.
- **Branch:** `ndbi021` (lightweight, minimal dependencies).
