"""Project-wide constants.

Every parameter that more than one stage needs lives here and is imported. No notebook
redefines a value, so there is exactly one place to change a decision and exactly one
place to read what the current decision is.

The temporal and cost constants below are the Stage 0 decisions. Changing any of them
invalidates work downstream, so read `docs/problem_definition.md` before editing.
"""

from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Named artifacts, so no stage spells a filename twice.
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"
RAW_PARQUET = RAW_DIR / "transactions_raw.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "customer_features.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
MODEL_SELECTION_JSON = MODELS_DIR / "model_selection.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

RANDOM_SEED = 42

# --------------------------------------------------------------------------------------
# Temporal design (Stage 0, section 5 of docs/problem_definition.md)
#
#   OBSERVATION WINDOW (features)              PREDICTION WINDOW (label)
#   2009-12-01 -------------------> 2011-09-11 -------------------> 2011-12-10
#                                       ^
#                                    CUTOFF
#
# Features use transactions strictly BEFORE the cutoff. The label uses transactions in
# [CUTOFF_DATE, PREDICTION_END). Recency is always measured against CUTOFF_DATE and never
# against max(invoice_date), which would silently import the future.
#
# The cutoff is set so that CUTOFF_DATE + HORIZON_DAYS clears the last timestamp in the
# dataset, keeping the prediction window fully populated. Verified in Stage 2: the data
# runs to 2011-12-09 12:50:00, so an exclusive bound of 2011-12-10 captures the final
# trading day. A cutoff of 2011-09-10 would have ended the window at midnight on
# 2011-12-09 and discarded 1,633 transactions, flipping 3 customer labels.
#
# These are datetimes rather than dates because `invoice_date` is a timestamp column;
# comparing like to like avoids dtype friction in Polars. Midnight is the boundary.
# --------------------------------------------------------------------------------------

OBSERVATION_START = datetime(2009, 12, 1)
CUTOFF_DATE = datetime(2011, 9, 11)
HORIZON_DAYS = 90
PREDICTION_END = CUTOFF_DATE + timedelta(days=HORIZON_DAYS)

# Lookback windows used by the momentum features in Stage 5, all anchored to CUTOFF_DATE.
LOOKBACK_WINDOWS_DAYS = (30, 90, 180, 365)

# --------------------------------------------------------------------------------------
# Validation (Stage 0, section 6)
# --------------------------------------------------------------------------------------

TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------------------
# Decision layer cost model (Stage 0, section 3)
#
# Named by business meaning, not by confusion-matrix cell, because the polarity inverts:
# the model predicts P(repeat), but the campaign acts on the NEGATIVE class. So a missed
# churner is a false POSITIVE in standard label terms, and a wasted offer is a false
# negative. Naming these COST_FP / COST_FN would invite exactly the wrong wiring.
#
# Only the ratio matters. We normalize the recoverable value of a churner to 1.0 and
# assume the discount costs 15 percent of what it can recover.
# --------------------------------------------------------------------------------------

COST_MISSED_CHURNER = 1.0  # did not repeat, we predicted repeat, no offer sent
COST_WASTED_OFFER = 0.15  # would have repeated, we predicted otherwise, offer sent

# Minimum acceptable performance, fixed before modeling (Stage 0, section 3).
MIN_ACCEPTABLE_ROC_AUC = 0.75
MIN_COST_REDUCTION_VS_TRIVIAL = 0.20

# --------------------------------------------------------------------------------------
# Feature schema (Stage 5, notebooks/03_features_and_labels.ipynb section 6)
#
# Countries with at least MIN_COUNTRY_CUSTOMERS customers in the observation window, all
# others fold into "Other". Pinned once here and never inferred from a batch, so encoding
# one row and encoding a million rows produce identical columns. Covers 96.3 percent of
# customers directly.
# --------------------------------------------------------------------------------------

MIN_COUNTRY_CUSTOMERS = 20

COUNTRY_VOCAB = (
    "Belgium", "France", "Germany", "Netherlands", "Other", "Spain", "Switzerland",
    "United Kingdom",
)

# The exact, ordered column set of the encoded feature matrix, excluding `customer_id`
# and `y`. `encode_features()` in `src/features.py` reindexes to this list, which is what
# guarantees a single row and a full batch produce identical columns in identical order.
# One-hot country columns are named `country_<value>` and derived from COUNTRY_VOCAB
# rather than duplicated here, so the two can never drift apart.
FEATURE_COLUMNS = (
    # RFM core
    "recency_days", "frequency", "monetary_total", "monetary_avg_per_order",
    # lifecycle
    "tenure_days", "avg_days_between_orders", "recency_over_avg_gap",
    "is_single_order_customer",
    # basket
    "total_quantity", "distinct_products", "avg_items_per_order", "avg_unit_price",
    "avg_distinct_products_per_order", "distinct_active_months",
    # returns
    "cancel_order_count", "cancel_order_rate", "return_value_ratio",
    # momentum
    "orders_last_30d", "orders_last_90d", "orders_last_180d",
    "spend_last_90d", "spend_last_365d", "spend_momentum",
) + tuple(f"country_{c}" for c in COUNTRY_VOCAB)
