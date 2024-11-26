# About
This is a repository with sources for the paper *Accuracy and Beyond-Accuracy Perspectives of Multi-Objective Recommender Systems* submitted to Information Processing & Management journal. Source codes are based on the [EasyStudy framework](https://github.com/pdokoupil/EasyStudy/tree/feature/recsys2023) enhanced by newly implemented plugin that describes overall flow of the user study.

# Authors
- Patrik Dokoupil
- Ludovico Boratto
- Ladislav Peska

# Contacts
- Patrik Dokoupil patrik.dokoupil@matfyz.cuni.cz
- Ladislav Peska ladislav.peska@matfyz.cuni.cz

# Reproducibility
Details on how to run EasyStudy and create instance of the user study are described in the [EasyStudy repository](https://github.com/pdokoupil/EasyStudy/blob/feature/recsys2023/README.md).

The particular plugin from which the study should be created is called `journal`, with the following parameters:
- Data loader: `Goodbooks-10k dataset` or `Movielens Genome 2021 dataset`
- Diversity metric: `CF-ILD`
- Diversity metric for compare-alphas: `SELECTED`
- Following "About" text override: `<h3>About</h3>\n<p>This study aims to evaluate multi-objective recommender systems and how users interact with them. The study is expected to be completed in around 20 to 30 minutes.</p>`
- Other parameters could be kept as is.