# E-commerce Repeat Purchase Prediction

Predict whether an existing customer will place another order in the next 90 days, so a
retention team can send win-back offers to the customers about to go quiet.

Built on the UCI **Online Retail II** transaction log (a UK online gift retailer,
2009-12-01 to 2011-12-09). The arc of the project is transactional data, to customer
behavior, to feature engineering, to supervised machine learning, to a final decision.

## Conclusion

**The project recommends against deploying a model.** XGBoost, the best candidate
found, ranks customers better than the incumbent RFM heuristic in a way that survives
cross-validation, but the resulting reduction in expected campaign cost (3.9 percent)
falls well short of the 20 percent bar set before any modeling began. Full reasoning in
[docs/model_selection.md](docs/model_selection.md); the machine-readable record is in
`models/model_selection.json`.

## Problem in one line

Binary classification at customer grain. Features come from the 21 months before
2011-09-11; the label is whether the customer ordered again in the 90 days after it.

Full framing, including the cost model that drove the final decision, is in
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
uv run ruff check .                # lint
```

Never call bare `python` or `pip`. Everything goes through `uv run` / `uv add` so the
locked environment is the only environment.

## Layout

```
data/raw/          untouched source data
data/processed/    cleaned tables, feature matrix, the saved train/test split
data/output/       out-of-fold prediction files used for threshold tuning
notebooks/         numbered, one per stage; the narrative of the project
src/               all reusable logic (cleaning, features, labels, modeling)
models/            fitted pipelines and the final selection record
docs/              problem definition, provenance, and the final decision doc
```

The split of responsibility matters: **notebooks tell the story, `src/` holds the
logic.** Every transform used more than once lives in `src/` and is imported into the
notebook, never copy-pasted.

## Notebooks

| Notebook | Stage | Produces |
|---|---|---|
| `01_eda.ipynb` | EDA and leakage audit | drop list, candidate features |
| `02_cleaning.ipynb` | Cleaning | `data/processed/transactions_clean.parquet` |
| `03_features_and_labels.ipynb` | Features and labels | `data/processed/customer_features.parquet` |
| `04_split_and_baseline.ipynb` | Split, scale, baselines | `train.parquet`, `test.parquet`, `models/baseline_logreg.joblib` |
| `05_advanced_models.ipynb` | XGBoost and LightGBM | `models/xgboost.joblib`, `models/lightgbm.joblib` |
| `06_comparison_and_threshold.ipynb` | Comparison, threshold, final decision | `models/model_selection.json` |

## Progress

See the stage table and key decisions in [CLAUDE.md](CLAUDE.md), following
[ML_WORKFLOW.md](ML_WORKFLOW.md).

| Stage | Status |
|---|---|
| 0 to 8 | done, project concluded at Stage 8 by decision |
| 9 to 11 (explainability, final export, serving) | out of scope |
