# E-commerce Repeat Purchase Prediction

Predict whether an existing customer will place another order in the next 90 days, so the
retention team can send win-back offers to the ones about to go quiet.

**Read `docs/problem_definition.md` before touching anything.** It is the Stage 0
document and every later stage must trace back to it. Read `ML_WORKFLOW.md` for the
process and `work-with-jupyter.md` for the notebook conventions.

---

## Stage table

Work proceeds **one stage per session turn**. Finish a stage's exit check, report, and
stop for user review before starting the next. Never build ahead.

| Stage | Deliverable | Status |
|---|---|---|
| 0 | Problem definition and task framing | **DONE** (`docs/problem_definition.md`) |
| 1 | Project scaffold: uv env, folders, `src/logger.py`, `src/config.py`, git | **DONE** |
| 2 | Data acquisition: `src/data.py`, `docs/data_provenance.md`, `data/raw/` | **DONE** |
| 3 | EDA and leakage audit: `notebooks/01_eda.ipynb` | pending |
| 4 | Cleaning: `src/cleaning.py`, `notebooks/02_cleaning.ipynb` | pending |
| 5 | Features and labels: `src/features.py`, `src/labels.py`, `notebooks/03_features_and_labels.ipynb` | pending |
| 6 | Split and baselines: `notebooks/04_split_and_baseline.ipynb` | pending |
| 7 | Advanced models: `notebooks/05_advanced_models.ipynb` | pending |
| 8 | Comparison, threshold, selection: `notebooks/06_comparison_and_threshold.ipynb` | pending |
| 9 | Explainability and error analysis: `notebooks/07_explainability_and_errors.ipynb` | pending |
| 10 | Final model export: `notebooks/08_final_export.ipynb` | pending |
| 11 | FastAPI inference plus `tests/` | pending |

---

## Key decisions

Append to this list as stages complete. Never relitigate an entry without new evidence,
and read `docs/` first.

### Stage 0

| # | Decision | Rationale |
|---|---|---|
| 0.1 | Binary classification at customer grain, positive class = "will repeat" | Matches the project question and makes `predict_proba[:, 1]` mean the intuitive thing |
| 0.2 | Cutoff 2011-09-11, horizon 90 days, observation window from 2009-12-01 | Maximizes feature history (about 21 months) while leaving an actionable 3-month horizon. Cutoff set so cutoff + 90d clears the dataset's last timestamp, keeping the label window fully populated. Corrected in Stage 2, see decision 2.3 |
| 0.3 | Population = non-null `Customer ID` with at least one non-cancelled invoice before cutoff | A customer who first appears after the cutoff did not exist to us at scoring time |
| 0.4 | Recency anchored to `CUTOFF_DATE`, never to `max(invoice_date)` | Anchoring to the data max silently imports the future. The primary leak risk in this project |
| 0.5 | Random stratified 80/20 split, StratifiedKFold(5) on train | One row per customer, no time axis and no repeated entity inside the table, so temporal and group splits are unnecessary |
| 0.6 | Selection on out-of-fold CV ROC-AUC, never on the test table | Comparing test scores across candidates is selection on the test set |
| 0.7 | Threshold tuned by expected-cost minimization at `COST_MISSED_CHURNER : COST_WASTED_OFFER = 1.0 : 0.15` | Missing a churner costs far more than a wasted discount. 0.5 is not a business decision |
| 0.8 | **No `class_weight`, no SMOTE, no resampling anywhere** | One knob per problem. Threshold tuning is the chosen imbalance mechanism; stacking both makes neither measurable |
| 0.9 | RFM quintile heuristic is a required baseline, not just an analysis step | Beating the majority class proves nothing. Beating the rule marketing already uses is the real bar |
| 0.10 | Seasonality accepted as a documented limitation | The 90-day window covers the Christmas peak. Data ends 2011-12-09, so no cutoff avoids it |

### Stage 1

| # | Decision | Rationale |
|---|---|---|
| 1.1 | Python pinned to `>=3.12,<3.13` | LightGBM, XGBoost and SHAP wheels lag new runtimes. Verified: all import cleanly on 3.12.13 |
| 1.2 | `fastexcel` instead of `openpyxl` for the source xlsx | It is the engine Polars uses natively for `read_excel`, so the Excel read stays in Polars rather than routing through pandas |
| 1.3 | Flat layout (`src/` as a plain package, not installed) | Matches the `sys.path.append(PROJECT_ROOT)` notebook convention in `work-with-jupyter.md`. `uv init --bare` avoids a packaging build step nothing needs |
| 1.4 | `setup_logger` reuses an existing handler instead of adding one | Re-running a notebook cell would otherwise attach a second handler and print every line twice. `propagate = False` for the same reason |
| 1.5 | `COUNTRY_VOCAB` and `FEATURE_COLUMNS` left undefined until Stage 5 | An early import should fail loudly rather than silently encode against `None` or an empty vocabulary |
| 1.6 | Cost constants named `COST_MISSED_CHURNER` / `COST_WASTED_OFFER`, not `COST_FP` / `COST_FN` | The class polarity inverts (see decision 0.1). Confusion-matrix names would invite exactly the wrong wiring at Stage 8 |

### Stage 2

| # | Decision | Rationale |
|---|---|---|
| 2.1 | **`SOURCE_SCHEMA` pins all eight dtypes; the Excel reader never infers** | Inference sampled the opening rows where every `Invoice` looks numeric, chose Int64, and silently nulled all 19,500 `C`-prefixed cancellation invoices. No error, correct row count, invisible damage. Would have flattened `is_cancellation` to always-false and corrupted `frequency`, the top RFM feature |
| 2.2 | `validate_raw()` runs on every load and raises on null invoices or zero cancellations | Regression guard for 2.1. The bug's failure mode was silence, so the guard has to be an assertion, not a log line |
| 2.3 | **`CUTOFF_DATE` moved 2011-09-10 to 2011-09-11** | `max(invoice_date)` is 2011-12-09 12:50. The old exclusive midnight bound discarded 1,633 transactions across 39 customers, wrongly flipping 3 labels to non-repeat. Now `PREDICTION_END` = 2011-12-10 and nothing is lost |
| 2.4 | Raw Parquet is a format conversion only, no type coercion or filtering | Keeps `data/raw/` faithful to source. All cleaning belongs to Stage 4, so there is one place that owns it |
| 2.5 | Base rate is about 43 percent, so imbalance is not the central problem | Threshold tuning in Stage 8 is justified by **asymmetric cost**, not by skew. Reinforces decision 0.8: still no class weighting |

---

## Conventions

- **uv for everything.** `uv add <dep>`, `uv run <script>`. Never bare `pip` or `python`.
- **Polars** for data work in Stages 2 through 5. Convert to pandas only at the sklearn
  boundary in Stage 6, where `ColumnTransformer` and SHAP need named columns.
- **`logger.info`, never `print`**, in notebooks and modules alike:
  `from src.logger import setup_logger`.
- **Notebooks tell the story, `src/` holds the logic.** Anything the API will run at
  inference time lives in `src/` and is imported into the notebook, never copy-pasted.
- **Append notebook cells at the bottom only.** Never insert mid-notebook.
- **Constants live in `src/config.py`** and are imported. No notebook redefines a
  parameter.
- **Never use em dashes** in any writing for this project: docs, markdown cells, comments,
  commit messages. Use commas, colons, parentheses or separate sentences.

### Notebook standard

Every notebook from Stage 3 onward follows this shape:

1. Title cell: stage number, purpose in two sentences, what the reader should conclude.
2. Constants cell: imported from `src/config.py`, at the very top.
3. Setup cell: the `PROJECT_ROOT` / `sys.path` / `DATA_DIR` pattern plus `setup_logger`.
4. Section markdown cells (`## Load Data` and so on) opening each major block.

Around every code cell, a markdown sandwich:

- **Before:** what it does, why it is needed here, and **what result is expected**.
  Stating the expectation up front is what makes a surprising result visible instead of
  quietly accepted.
- **After:** what actually came out, in plain language, and what it changes for the next
  step. Where the outcome contradicts the expectation, say so explicitly.

Plus:

- Every filter or imputation shows before and after counts, and the share of rows and
  revenue affected. A drop with no number attached is not a documented drop.
- Every save is followed by a reload-and-verify cell: assert shape, assert no nulls where
  none are allowed, assert dtypes, spot-check a row.
- Every notebook closes with a summary cell: what was produced, key numbers, decisions and
  rationale, open questions inherited by the next stage. This cell is the stage exit check
  written out.
- Execute end to end and confirm no cell raised before reporting a stage done.

---

## The three rules most easily broken here

1. **The test set is sacred.** Split once, save to disk, load those files everywhere. Tune
   hyperparameters *and* thresholds *and* model choice on out-of-fold training data. The
   test set confirms a decision already made.
2. **No post-cutoff data in any feature.** Features from `[2009-12-01, 2011-09-11)`, label
   from `[2011-09-11, 2011-12-10)`. Recency against `CUTOFF_DATE`.
3. **Train and serve run the same code.** Pin `COUNTRY_VOCAB` explicitly, bundle scalers
   inside the pipeline, reindex serving input to the schema in `model_metadata.json`. One
   row and a million rows must encode identically, and there is a pytest for it.
