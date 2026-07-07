# Create a study (no code)

This walks through the Administration UI to create and run a study **without writing any code**, using
the built-in `fastcompare` template.

## Prerequisites
- EasyStudy running (see [Quickstart](quickstart.md)) and datasets fetched.
- An admin account (`/signup` → `/login`).

## Step 1 — start a new fastcompare study
From Administration, click **Create** under *fastcompare*. You get a parameter form; every field has a
`?` help tooltip.

## Step 2 — choose data & elicitation
- **Data loader** — the domain/dataset participants will see (e.g. *Goodbooks-10k*, *Filtered ML-25M*).
- **Preference elicitation** — how you gather initial signal to personalize. *Popularity Sampling*
  works in the lightweight core and needs no extras.

## Step 3 — choose algorithms to compare
- Pick **2–3 algorithms** (e.g. **EASE** + a baseline). Their **parameters appear as editable fields**
  automatically (this is the annotation-driven UI — e.g. EASE exposes `l2` and `positive_threshold`).
- Only algorithms whose dependencies are installed appear. Install `easystudy[tensorflow]` /
  `[lenskit]` to unlock more.

## Step 4 — configure the flow
- **Recommendation size (K)** — items shown per algorithm per round.
- **Number of iterations (N)** — how many comparison rounds.
- **Layout** — rows vs columns (affects presentation/position bias).
- **Prompts / texts** — the wording participants see.
- **Prolific code** (optional) — if recruiting via Prolific.

## Step 5 — create & wait for initialization
Submit. The study **initializes in the background** (training the algorithms on the dataset). When it
finishes it flips to **active** and shows a **join URL**. If something fails, the error surfaces in the
admin UI (`initialization_error`).

## Step 6 — distribute the join URL
Share the join URL with participants (or paste it into Prolific/e-mail). Each participant goes through
consent → elicitation → N rounds → final page/questionnaire.

## Step 7 — monitor & collect
Watch progress and open **Results** for built-in metrics, or query the database directly
([Data model](data-model.md)).

## Creating variants
Create **multiple instances** from the same template with different parameters (e.g. rows vs columns,
different algorithms) — each gets its own URL and its own data. This is the standard way to run
between-subject conditions today.

!!! tip "Coming in v2"
    A drag-and-drop **no-code study designer** (interleave questionnaires, attention checks, branching)
    and a **template gallery** are the headline v2 features — this page will grow with them.
