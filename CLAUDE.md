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
| 3 | EDA and leakage audit: `notebooks/01_eda.ipynb` | **DONE** |
| 4 | Cleaning: `src/cleaning.py`, `notebooks/02_cleaning.ipynb` | pending |
| 5 | Features and labels: `src/features.py`, `src/labels.py`, `notebooks/03_features_and_labels.ipynb` | pending |
| 6 | Split and baselines: `notebooks/04_split_and_baseline.ipynb` | pending |
| 7 | Advanced models (XGBoost, LightGBM): `notebooks/05_advanced_models.ipynb` | pending |
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

### Stage 3

Full evidence in `notebooks/01_eda.ipynb`. Numbers below are from the executed notebook.

| # | Decision | Rationale |
|---|---|---|
| 3.1 | **Stage 4 drops non-product rows by an explicit code list, never by regex** | `DCGS*` and `SP1002` look like junk codes but are **real products** (`MISO PRETTY GUM`, `SUNJAR LED NIGHT LIGHT`, `KID'S CHALKBOARD/EASEL`). A "drop everything non-standard" rule would silently delete real purchases |
| 3.2 | Negative quantity is **not** a cancellation marker | 3,457 rows have negative quantity and no `C` prefix. All zero price, all unattributed, described as `damages`, `check`, `missing`, `smashed`. These are warehouse write-offs, not customer returns |
| 3.3 | Three invoice prefixes exist, not two: numeric, `C`, and `A` | The 6 `A` rows are "Adjust bad debt" accounting entries carrying -147,614 revenue. Undocumented; found by chasing the negative prices. Dropped in Stage 4 |
| 3.4 | Keep the 12,133 exact duplicate rows (1.14 percent) | Genuinely ambiguous (repeated till scans vs export fault). They do not change invoice-level features at all, and the only exposure is monetary totals. Revisit if monetary features misbehave |
| 3.5 | Do **not** clip quantity outliers | The extremes are a matched pair: invoice 581483 (+80,995) and its exact cancellation `C581484` (-80,995). Real orders, not errors. Netting handles them |
| 3.6 | Drop `description` and `source_sheet` columns | `description` duplicates `stock_code` and holds warehouse notes (`?`, `MIA`, `wet`). `source_sheet` encodes trading year, a crude time proxy and a potential leak |
| 3.7 | Unattributed rows leave in **Stage 5 as a population filter**, not in Stage 4 cleaning | 243,007 rows (22.8 percent) but only 13.7 percent of revenue. Zero invoices mix attributed and unattributed rows, so aggregation stays clean. Population definition and cleaning are different jobs |
| 3.8 | **Build `cancel_order_rate`, not just the raw count** | Cancellations correlate **positively** with repeat (0.577 vs 0.329), the opposite of the naive expectation. Cancelling proxies for order volume. The rate lets the model separate "returns a lot relative to purchases" from "simply buys a lot" |
| 3.9 | Tenure is a denominator, not a predictor | `tenure_days` correlates +0.044 with the label, essentially nothing, while `recency_days` correlates -0.437. How long someone has been a customer says little; how recently they bought says a lot. Hence `recency_over_avg_gap` |
| 3.10 | **The RFM heuristic is a serious competitor, not a formality** | Combined RFM score separates repeat rate from 0.072 to 0.906. A model reaching 0.80 ROC-AUC may not beat it. Stage 6 must report it honestly |
| 3.11 | No Saturday trading (402 rows in two years) | Explains the day-and-a-half gap at the window boundary (2011-09-10 is a Saturday, 0 rows). Day-count features carry a weekly rhythm, so do not build finer-grained time features |

### Pre-registered for Stage 7

Decided ahead of the stage so the model roster is fixed before any results are seen.
Choosing candidates after seeing scores is how a comparison stops being a comparison.

| # | Decision | Rationale |
|---|---|---|
| 7.1 | **Model roster is Logistic Regression, XGBoost, LightGBM. Random Forest is dropped.** | User decision, 2026-08-16: computation cost. RF is the weakest earner of the three tree candidates here, since XGBoost and LightGBM cover the same bagged-tree ground with better accuracy per unit of compute. Nothing in the comparison is lost that the two boosters do not already provide |
| 7.2 | The two boosters still count as two independent families for the comparison | Principle: the workflow asks for two or three families tuned on identical folds. XGBoost and LightGBM differ enough in growth policy and regularization to be a real comparison, with Logistic Regression as the interpretable third |

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
2. Setup cell: the `PROJECT_ROOT` / `sys.path` / `DATA_DIR` pattern. This must come before
   the constants cell, because `sys.path` has to be extended before anything can import
   from `src`.
3. Constants cell: imported from `src/config.py`, plus `setup_logger`. Nothing magic
   appears further down.
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
