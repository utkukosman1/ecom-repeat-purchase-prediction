# Data Provenance

Stage 2 of `ML_WORKFLOW.md`. Raw data lands in `data/raw/` and is never modified in place.

## Source

| Property | Value |
|---|---|
| Name | Online Retail II |
| Repository | UCI Machine Learning Repository, dataset ID 502 |
| URL | `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip` |
| Downloaded | 2026-08-16 |
| Archive size | 43.5 MB (zip), containing `online_retail_II.xlsx` |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Description | Transaction log of a UK-based online gift retailer selling mostly to wholesale customers |

## Acquisition

Scripted, not manual:

```bash
uv run python -m src.data
```

`src/data.py` downloads the zip, extracts the workbook to `data/raw/online_retail_II.xlsx`,
reads both sheets, and writes `data/raw/transactions_raw.parquet`. The download is skipped
when the workbook is already present. The Parquet file exists because parsing 1M rows of
xlsx takes about 15 seconds per sheet and every later stage would otherwise pay it.

The Parquet is a **format conversion only**: columns renamed to snake_case, a
`source_sheet` column added so per-year counts stay checkable. No filtering, no derived
columns, no imputation. Those belong to Stage 4.

## Contents as loaded

| Property | Value |
|---|---|
| Rows | 1,067,371 |
| Columns | 9 (8 source columns plus `source_sheet`) |
| Sheet `Year 2009-2010` | 525,461 rows |
| Sheet `Year 2010-2011` | 541,910 rows |
| Date range | 2009-12-01 07:45:00 to 2011-12-09 12:50:00 |
| Distinct invoices | 53,628 |
| Distinct customers | 5,943 (excluding nulls) |
| Distinct countries | 43 |
| Rows with null `customer_id` | 243,007 (22.8 percent) |
| Cancellation rows (`C`-prefixed invoice) | 19,494 |
| Grain | One row per invoice line item |

### Split across the Stage 0 windows

Verified after the cutoff correction below:

| Window | Range | Rows | Share |
|---|---|---|---|
| Observation (features) | 2009-12-01 to 2011-09-11 | 859,515 | 80.5 percent |
| Prediction (label) | 2011-09-11 to 2011-12-10 | 207,856 | 19.5 percent |
| Beyond `PREDICTION_END` | none | 0 | 0 |

Indicative modeling population, confirmed properly in Stage 5: **5,346 customers** in the
observation window, of whom 2,322 transact again in the prediction window, an approximate
base rate of **43.4 percent**. Close to balanced, which means class imbalance is not the
central difficulty here and the threshold work in Stage 8 is driven by asymmetric cost
rather than by skew.

Worth noting early: roughly 5,300 rows is a small modeling table. Cross-validation
variance will be visible, and hyperparameter search has limited room to help. That is a
constraint to respect in Stages 7 and 8, not something to fix.

### Schema

| Column | Type | Note |
|---|---|---|
| `invoice` | String | `C` prefix marks a cancellation. **Must not be numeric** |
| `stock_code` | String | Mixes product codes such as `85123A` with non-product entries such as `POST`, `D`, `M`, `ADJUST` |
| `description` | String | Free text, has nulls |
| `quantity` | Int64 | Negative on cancellations and returns |
| `invoice_date` | Datetime (ms) | Timestamp, not a date |
| `price` | Float64 | Unit price in GBP |
| `customer_id` | Int64 | Nullable. 22.8 percent of rows are unattributed |
| `country` | String | Customer country, UK-dominated |
| `source_sheet` | String | Added at load, not a source column |

## Findings recorded at acquisition

### 1. Dtype inference silently destroyed every cancellation invoice

The first load let the Excel reader infer dtypes. It sampled the opening rows of each
sheet, where every `Invoice` value looks numeric, chose `Int64`, and then **nulled 19,500
rows** whose invoice carried the `C` cancellation prefix.

The failure was invisible: row counts were correct, no error was raised, and the frame
looked healthy. The damage would have surfaced much later as two corrupted features:

- `is_cancellation`, derived from the `C` prefix in Stage 4, would have been uniformly
  false.
- `frequency`, the distinct invoice count and the most important RFM feature, would have
  collapsed 19,500 rows into a single null invoice group.

**Resolution:** `SOURCE_SCHEMA` in `src/data.py` pins all eight column types explicitly.
`validate_raw()` runs on every load and raises if any invoice is null or if no
`C`-prefixed rows are present. The number of distinct invoices recovered from 24,222 to
28,816 on sheet one alone.

This is the same principle the workflow applies to categorical vocabularies, one step
earlier in the pipeline: **pin the schema, never let a reader infer it from whatever
sample it happens to see.**

### 2. The prediction window boundary was one day short

`max(invoice_date)` is `2011-12-09 12:50:00`, so the dataset's final trading day is
populated until midday.

With the previous `CUTOFF_DATE` of 2011-09-10, `PREDICTION_END` fell at midnight on
2011-12-09, and the exclusive bound discarded **1,633 transactions across 39 customers**.
Three of those customers appear *only* in the discarded tail, so three labels would have
been wrongly recorded as non-repeat.

**Resolution:** `CUTOFF_DATE` moved to **2011-09-11**, putting `PREDICTION_END` at
2011-12-10 and capturing the full final day. The observation window loses one day, which
is immaterial against 21 months.

Three labels out of roughly four thousand is a small effect. It was fixed because it was
free to fix and because a label definition that quietly drops the boundary day is the kind
of detail that undermines trust in every number downstream.

## Reproducibility

`download_raw()` is idempotent and skips an existing workbook. To force a clean re-fetch:

```python
from src.data import download_raw, build_raw_parquet
download_raw(force=True)
build_raw_parquet(force=True)
```

Neither `data/raw/` nor `models/` is tracked in git. A clean clone reproduces both from
the command above.
