"""Customer-level feature engineering for the repeat-purchase model.

`build_customer_features` turns cleaned transactions into one row per customer, using
only transactions strictly before `cutoff`. `encode_features` one-hots the pinned country
vocabulary and reindexes to `FEATURE_COLUMNS`, so a single row and a full batch produce
identical columns; this is what the FastAPI serving layer calls at inference time.

Prototyped and verified cell by cell in `notebooks/03_features_and_labels.ipynb`; see that
notebook for the reasoning behind each feature and every intermediate result.
"""

from datetime import timedelta

import polars as pl

from src.config import COUNTRY_VOCAB, CUTOFF_DATE, OBSERVATION_START
from src.labels import build_population
from src.logger import setup_logger

logger = setup_logger("features")

EPS = 1e-6


def _obs_purchases(tx: pl.DataFrame, cutoff) -> pl.DataFrame:
    """Non-cancelled, attributed, pre-cutoff rows: the base for every feature below."""
    return tx.filter(
        (pl.col("invoice_date") < cutoff)
        & ~pl.col("is_cancellation")
        & pl.col("customer_id").is_not_null()
    )


def _rfm_and_lifecycle(obs_purchases: pl.DataFrame, cutoff, window_days: int) -> pl.DataFrame:
    """Recency, frequency, monetary, tenure, and the gap-ratio features.

    `avg_days_between_orders` is undefined for a single-order customer. Rather than a
    null, they get `window_days` (the full observation span), the largest gap we can
    justify with no repeat history to judge from. `is_single_order_customer` lets the
    model separate "no gap history" from "a genuinely wide gap".

    Floored at 1.0 day for multi-order customers: `.dt.total_days()` truncates to whole
    days, so two or more orders placed on the same calendar day compute a gap of exactly
    0, and `recency_days / 0` in `recency_over_avg_gap` below produces `inf`. Not a null,
    so the Stage 5 null-count check did not catch it; found when the Stage 6 Logistic
    Regression pipeline rejected the matrix outright. 73 of 5,256 customers were affected.
    A 1-day floor matches the day-level granularity already chosen elsewhere (no Saturday
    trading means day-count features carry a weekly rhythm; see EDA decision 3.11) rather
    than resolving sub-day timing that nothing else in the feature set uses.
    """
    return (
        obs_purchases.group_by("customer_id")
        .agg(
            (cutoff - pl.col("invoice_date").max()).dt.total_days().alias("recency_days"),
            (cutoff - pl.col("invoice_date").min()).dt.total_days().alias("tenure_days"),
            pl.col("invoice").n_unique().alias("frequency"),
            pl.col("line_revenue").sum().alias("monetary_total"),
            pl.col("invoice_date").min().alias("_first_order"),
            pl.col("invoice_date").max().alias("_last_order"),
        )
        .with_columns(
            (pl.col("monetary_total") / pl.col("frequency")).alias("monetary_avg_per_order"),
            (pl.col("frequency") == 1).alias("is_single_order_customer"),
        )
        .with_columns(
            pl.when(pl.col("frequency") > 1)
            .then(
                pl.max_horizontal(
                    (pl.col("_last_order") - pl.col("_first_order")).dt.total_days()
                    / (pl.col("frequency") - 1),
                    pl.lit(1.0),
                )
            )
            .otherwise(pl.lit(float(window_days)))
            .alias("avg_days_between_orders")
        )
        .with_columns(
            (pl.col("recency_days") / pl.col("avg_days_between_orders"))
            .alias("recency_over_avg_gap")
        )
        .drop("_first_order", "_last_order")
    )


def _basket(obs_purchases: pl.DataFrame) -> pl.DataFrame:
    """Basket-size and product-variety features."""
    basket = obs_purchases.group_by("customer_id").agg(
        pl.col("quantity").sum().alias("total_quantity"),
        pl.col("stock_code").n_unique().alias("distinct_products"),
        (pl.col("quantity").sum() / pl.col("invoice").n_unique()).alias("avg_items_per_order"),
        pl.col("price").mean().alias("avg_unit_price"),
        pl.col("invoice_date").dt.strftime("%Y-%m").n_unique().alias("distinct_active_months"),
    )
    distinct_per_invoice = (
        obs_purchases.group_by(["customer_id", "invoice"])
        .agg(pl.col("stock_code").n_unique().alias("n_products"))
        .group_by("customer_id")
        .agg(pl.col("n_products").mean().alias("avg_distinct_products_per_order"))
    )
    return basket.join(distinct_per_invoice, on="customer_id", how="left")


def _returns(tx: pl.DataFrame, cutoff) -> pl.DataFrame:
    """Cancellation count and value, pre-cutoff, before dividing by frequency/monetary."""
    cancellations = tx.filter(
        (pl.col("invoice_date") < cutoff)
        & pl.col("is_cancellation")
        & pl.col("customer_id").is_not_null()
    )
    return cancellations.group_by("customer_id").agg(
        pl.col("invoice").n_unique().alias("cancel_order_count"),
        pl.col("line_revenue").sum().abs().alias("_cancel_value"),
    )


def _momentum(obs_purchases: pl.DataFrame, population: pl.DataFrame, cutoff) -> pl.DataFrame:
    """Order counts and spend inside fixed lookback windows before the cutoff."""
    momentum = population.clone()
    for days in (30, 90, 180):
        window_start = cutoff - timedelta(days=days)
        window = (
            obs_purchases.filter(pl.col("invoice_date") >= window_start)
            .group_by("customer_id")
            .agg(pl.col("invoice").n_unique().alias(f"orders_last_{days}d"))
        )
        momentum = momentum.join(window, on="customer_id", how="left").with_columns(
            pl.col(f"orders_last_{days}d").fill_null(0)
        )
    for days in (90, 365):
        window_start = cutoff - timedelta(days=days)
        window = (
            obs_purchases.filter(pl.col("invoice_date") >= window_start)
            .group_by("customer_id")
            .agg(pl.col("line_revenue").sum().alias(f"spend_last_{days}d"))
        )
        momentum = momentum.join(window, on="customer_id", how="left").with_columns(
            pl.col(f"spend_last_{days}d").fill_null(0.0)
        )
    return momentum.with_columns(
        (pl.col("spend_last_90d") / (pl.col("spend_last_365d") / 4 + EPS)).alias("spend_momentum")
    )


def _country(obs_purchases: pl.DataFrame) -> pl.DataFrame:
    """Most recent transaction's country, mapped to the pinned vocabulary plus Other.

    Most recent, not first or most frequent: that is what would actually be known about
    the customer at scoring time. 13 customers in the EDA appeared under more than one
    country, so this choice is deliberate, not incidental.
    """
    return (
        obs_purchases.sort("invoice_date")
        .group_by("customer_id")
        .agg(pl.col("country").last().alias("_raw_country"))
        .with_columns(
            pl.when(pl.col("_raw_country").is_in(COUNTRY_VOCAB))
            .then(pl.col("_raw_country"))
            .otherwise(pl.lit("Other"))
            .alias("country")
        )
        .drop("_raw_country")
    )


def build_customer_features(tx: pl.DataFrame, cutoff=CUTOFF_DATE) -> pl.DataFrame:
    """One row per customer: every raw (pre-encoding) feature, no nulls.

    Population is customers with at least one real purchase strictly before `cutoff`,
    matching `src.labels.build_population` exactly so features and labels always line up
    on the same customer set.
    """
    population = build_population(tx, cutoff)
    obs_purchases = _obs_purchases(tx, cutoff)
    window_days = (cutoff - OBSERVATION_START).days

    features = (
        population
        .join(_rfm_and_lifecycle(obs_purchases, cutoff, window_days), on="customer_id", how="left")
        .join(_basket(obs_purchases), on="customer_id", how="left")
        .join(_returns(tx, cutoff), on="customer_id", how="left")
        .join(_momentum(obs_purchases, population, cutoff), on="customer_id", how="left")
        .join(_country(obs_purchases), on="customer_id", how="left")
        .with_columns(
            pl.col("cancel_order_count").fill_null(0),
            pl.col("_cancel_value").fill_null(0.0),
        )
        .with_columns(
            (pl.col("cancel_order_count") / pl.col("frequency")).alias("cancel_order_rate"),
            (pl.col("_cancel_value") / pl.col("monetary_total")).alias("return_value_ratio"),
        )
        .drop("_cancel_value")
        .sort("customer_id")
    )

    null_report = {k: v for k, v in features.null_count().row(0, named=True).items() if v > 0}
    if null_report:
        raise ValueError(f"unexpected nulls in customer features: {null_report}")

    # Regression guard: a ratio feature (recency_over_avg_gap) once divided by a gap that
    # truncated to exactly 0, producing inf rather than null. A null-only check missed it;
    # this would not.
    numeric_cols = [c for c, dt in features.schema.items() if dt.is_numeric()]
    inf_report = {
        c: n for c in numeric_cols
        if (n := features.select(pl.col(c).is_infinite().sum()).item()) > 0
    }
    if inf_report:
        raise ValueError(f"non-finite values in customer features: {inf_report}")

    logger.info("built features: %s rows x %s columns", f"{features.height:,}", features.width)
    return features


def encode_features(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot `country` against the pinned vocabulary, cast booleans, order columns.

    Reindexes to `FEATURE_COLUMNS` so a single row and a full batch always produce the
    same column set in the same order, which is the property Stage 11 serving depends on.
    Imported lazily from `src.config` so this module can be imported before
    `FEATURE_COLUMNS` exists (it is set once, right after this function, at the bottom of
    `src/config.py`).
    """
    from src.config import FEATURE_COLUMNS

    encoded = df.with_columns(
        [(pl.col("country") == c).cast(pl.Int8).alias(f"country_{c}") for c in COUNTRY_VOCAB]
    ).drop("country").with_columns(
        pl.col("is_single_order_customer").cast(pl.Int8)
    )

    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded = encoded.with_columns(pl.lit(0, dtype=pl.Int8).alias(col))

    id_cols = [c for c in ("customer_id", "y") if c in encoded.columns]
    return encoded.select(id_cols + list(FEATURE_COLUMNS))
