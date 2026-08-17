"""Cleaning for the Online Retail II transaction log.

`clean_transactions` is the single function serving will also call, so the notebook and
the API run identical logic. Every step is the one prototyped and measured in
`notebooks/02_cleaning.ipynb`; see that notebook for the before/after counts and revenue
share each step removes.

Tolerant of a frame that never had `description` or `source_sheet` (a single inference
row will not), via `strict=False` on the drop.
"""

import polars as pl

from src.config import CLEAN_PARQUET, RAW_PARQUET
from src.logger import setup_logger

logger = setup_logger("cleaning")

# Columns whose whitespace must be trimmed before any code-based filter runs, since a
# trailing space (`47503J `) would otherwise fail an exact-match filter.
STRING_COLS = ("invoice", "stock_code", "description", "country")

# Explicit, not a regex. DCGS* and SP1002 look like junk codes but are real products
# (MISO PRETTY GUM, SUNJAR LED NIGHT LIGHT, KID'S CHALKBOARD/EASEL); a pattern rule would
# silently delete real purchases. See notebooks/01_eda.ipynb section 4.
NON_PRODUCT_CODES = frozenset({
    # shipping and carriage
    "POST", "DOT", "C2", "C3",
    # fees, adjustments, accounting
    "M", "m", "D", "S", "BANK CHARGES", "AMAZONFEE", "CRUK", "B", "ADJUST", "ADJUST2", "PADS",
    # internal test rows
    "TEST001", "TEST002",
    # gift vouchers: payment instruments, not products
    "GIFT", "gift_0001_10", "gift_0001_20", "gift_0001_30", "gift_0001_40", "gift_0001_50",
    "gift_0001_60", "gift_0001_70", "gift_0001_80", "gift_0001_90",
})


def clean_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """Clean a raw transaction frame into a typed, filtered table.

    Steps, in order (order matters: whitespace before code filters, price before the
    negative-quantity backstop):

    1. Trim whitespace on string columns.
    2. Derive `is_cancellation` from the `C` invoice prefix, not from `quantity < 0`.
    3. Drop `A`-prefixed invoices (bad-debt accounting, not transactions).
    4. Drop non-product `stock_code` rows.
    5. Drop non-positive `price` rows.
    6. Drop any remaining negative-quantity row that is not a cancellation (backstop;
       expected to remove nothing given step 5, see the notebook).
    7. Cast `customer_id` to Int64, add `line_revenue`, drop `description` and
       `source_sheet`.

    Null `customer_id` rows are deliberately kept. Population filtering is a Stage 5
    concern, not a cleaning rule.
    """
    before_rows = df.height

    df = df.with_columns(
        [pl.col(c).str.strip_chars().alias(c) for c in STRING_COLS if c in df.columns]
    ).with_columns(
        pl.col("invoice").str.starts_with("C").alias("is_cancellation")
    )

    df = df.filter(~pl.col("invoice").str.starts_with("A"))
    df = df.filter(~pl.col("stock_code").is_in(NON_PRODUCT_CODES))
    df = df.filter(pl.col("price") > 0)
    df = df.filter(pl.col("quantity").gt(0) | pl.col("is_cancellation"))

    df = df.with_columns(
        pl.col("customer_id").cast(pl.Int64),
        (pl.col("quantity") * pl.col("price")).alias("line_revenue"),
    ).drop(["description", "source_sheet"], strict=False)

    logger.info(
        "cleaned: %s -> %s rows (-%.2f%%)",
        f"{before_rows:,}", f"{df.height:,}", 100 * (before_rows - df.height) / before_rows,
    )
    validate_clean(df)
    return df


def validate_clean(df: pl.DataFrame) -> None:
    """Guard the invariants clean_transactions is supposed to establish."""
    if df.filter(pl.col("invoice").str.starts_with("A")).height:
        raise ValueError("A-prefixed invoices survived cleaning")
    if df.filter(pl.col("price") <= 0).height:
        raise ValueError("non-positive price rows survived cleaning")
    if df.filter((pl.col("quantity") < 0) != pl.col("is_cancellation")).height:
        raise ValueError("negative quantity and is_cancellation disagree after cleaning")
    for col in ("invoice", "stock_code", "quantity", "price", "line_revenue"):
        if df[col].null_count():
            raise ValueError(f"unexpected nulls in {col} after cleaning")
    logger.info("validated: invariants hold on %s rows", f"{df.height:,}")


if __name__ == "__main__":
    raw = pl.read_parquet(RAW_PARQUET)
    clean = clean_transactions(raw)
    clean.write_parquet(CLEAN_PARQUET)
    logger.info("wrote %s (%.1f MB)", CLEAN_PARQUET.name, CLEAN_PARQUET.stat().st_size / 1024**2)
