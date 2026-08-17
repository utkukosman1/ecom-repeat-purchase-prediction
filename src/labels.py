"""Population and label construction for the repeat-purchase target.

Prototyped and verified in `notebooks/03_features_and_labels.ipynb` section 2. Population
and label are built from the same underlying rule (a non-cancelled purchase in a window),
applied to two disjoint windows, so the two functions below share one filter helper.
"""

import polars as pl

from src.config import CUTOFF_DATE, PREDICTION_END
from src.logger import setup_logger

logger = setup_logger("labels")


def _real_purchases(tx: pl.DataFrame, start, end) -> pl.DataFrame:
    """Non-cancelled, attributed transaction rows in `[start, end)`."""
    filters = [~pl.col("is_cancellation"), pl.col("customer_id").is_not_null()]
    if start is not None:
        filters.append(pl.col("invoice_date") >= start)
    if end is not None:
        filters.append(pl.col("invoice_date") < end)
    return tx.filter(pl.all_horizontal(filters))


def build_population(tx: pl.DataFrame, cutoff=CUTOFF_DATE) -> pl.DataFrame:
    """Customers with at least one real purchase strictly before `cutoff`.

    A customer whose only pre-cutoff activity is a cancellation has nothing to compute
    recency or frequency from, so they are excluded here rather than kept with broken
    features.
    """
    return _real_purchases(tx, None, cutoff).select("customer_id").unique()


def build_labels(
    tx: pl.DataFrame, cutoff=CUTOFF_DATE, prediction_end=PREDICTION_END
) -> pl.DataFrame:
    """One row per population customer: `customer_id`, `y`.

    `y = 1` if the customer has at least one real purchase in `[cutoff, prediction_end)`.
    """
    population = build_population(tx, cutoff)
    repeaters = (
        _real_purchases(tx, cutoff, prediction_end)
        .select("customer_id").unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    )
    labels = (
        population.join(repeaters, on="customer_id", how="left")
        .with_columns(pl.col("y").fill_null(0).cast(pl.Int8))
        .sort("customer_id")
    )
    logger.info(
        "population: %s customers, base rate: %.4f",
        f"{labels.height:,}", labels["y"].mean(),
    )
    return labels
