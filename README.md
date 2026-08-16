# E-commerce Repeat Purchase Prediction

Predict whether an existing customer will place another order in the next 90 days, so a
retention team can send win-back offers to the customers about to go quiet.

Built on the UCI **Online Retail II** transaction log (a UK online gift retailer,
2009-12-01 to 2011-12-09). The arc of the project is transactional data, to customer
behavior, to feature engineering, to supervised machine learning, to a served model.

## Problem in one line

Binary classification at customer grain. Features come from the 21 months before
2011-09-11; the label is whether the customer ordered again in the 90 days after it.

Full framing, including the cost model that sets the decision threshold, is in
[docs/problem_definition.md](docs/problem_definition.md). Read it first.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch the interpreter
if it is missing).

```bash
uv sync
```

That creates `.venv/` and installs everything from `uv.lock`.

## Getting the data

```bash
uv run python -m src.data
```

Downloads the UCI Online Retail II workbook (43.5 MB) to `data/raw/`, then writes
`data/raw/transactions_raw.parquet` (1,067,371 rows) so later stages skip the slow Excel
parse. The download is skipped if the workbook is already present.

Raw files are never modified in place. Provenance, schema and the two data issues found at
acquisition are documented in [docs/data_provenance.md](docs/data_provenance.md).

## Running things

```bash
uv run jupyter lab                 # explore and run notebooks
uv run pytest tests/ -v            # test suite (Stage 11)
uv run ruff check .                # lint
```

Never call bare `python` or `pip`. Everything goes through `uv run` / `uv add` so the
locked environment is the only environment.

## Layout

```
data/raw/          untouched source data
data/processed/    cleaned tables, feature matrix, the saved train/test split
data/output/       intermediate artifacts and diagnostics
notebooks/         numbered, one per stage; the narrative of the project
src/               all reusable logic; anything serving runs at inference time
models/            fitted pipelines, selection record, final artifact and metadata
docs/              problem definition and decision records
app/               FastAPI inference service (Stage 11)
tests/             pytest suite (Stage 11)
```

The split of responsibility matters: **notebooks tell the story, `src/` holds the
logic.** Anything the API will execute lives in `src/` and is imported into the notebook,
never copy-pasted, so training and serving cannot drift apart.

## Progress

See the stage table in [CLAUDE.md](CLAUDE.md). Stages are completed one at a time, each
with an exit check, following [ML_WORKFLOW.md](ML_WORKFLOW.md).

| Stage | Status |
|---|---|
| 0 Problem definition | done |
| 1 Scaffold | done |
| 2 Data acquisition | done |
| 3 EDA and leakage audit | done |
| 4 Cleaning | next |
| 5 to 11 | pending |
