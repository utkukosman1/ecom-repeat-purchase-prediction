"""Data acquisition for the UCI Online Retail II dataset.

Acquisition is scripted rather than manual so that a clean clone can reproduce
`data/raw/` with one command:

    uv run python -m src.data

The downloaded xlsx is the raw artifact and is never modified in place. `load_raw`
performs a format conversion only (Excel to Parquet, plus column renaming), because
parsing 1M rows of xlsx takes about a minute and every later stage would otherwise pay
that cost. Type coercion, filtering and derived columns belong to Stage 4 cleaning, not
here, so that `data/raw/` stays a faithful copy of the source.
"""

import urllib.request
import zipfile
from pathlib import Path

import polars as pl

from src.config import RAW_DIR, RAW_PARQUET, RAW_XLSX
from src.logger import setup_logger

logger = setup_logger("data")

# UCI Machine Learning Repository, dataset 502.
SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# The workbook splits the transaction log across two sheets, one per trading year.
SHEET_NAMES = ("Year 2009-2010", "Year 2010-2011")

# Explicit rather than a generic regex: there are only eight columns, and an explicit map
# fails loudly if the upstream schema ever changes, instead of silently renaming to
# something the rest of the project does not expect.
COLUMN_RENAMES = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# Pinned explicitly rather than inferred, and this is not a stylistic preference.
# The Excel reader infers dtypes from a sample of the first rows, where every Invoice
# value looks numeric. It therefore chose Int64 and silently nulled the 19,500
# C-prefixed cancellation invoices that appear later in the sheet, which would have
# destroyed the cancellation signal and corrupted `frequency` (distinct invoice count),
# the single most important RFM feature. Pinning the schema makes that impossible.
SOURCE_SCHEMA = {
    "Invoice": pl.String,  # C-prefixed for cancellations, so never numeric
    "StockCode": pl.String,  # mixes 85123A style codes with POST, D, M, ADJUST
    "Description": pl.String,
    "Quantity": pl.Int64,
    "InvoiceDate": pl.Datetime,
    "Price": pl.Float64,
    "Customer ID": pl.Int64,
    "Country": pl.String,
}


def download_raw(force: bool = False) -> Path:
    """Download and unzip the source workbook into `data/raw/`.

    Skips the download when the xlsx is already present, so re-running the module is
    cheap. Pass `force=True` to re-fetch.
    """
    if RAW_XLSX.exists() and not force:
        size_mb = RAW_XLSX.stat().st_size / 1024**2
        logger.info("raw workbook already present (%.1f MB), skipping download", size_mb)
        return RAW_XLSX

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "online_retail_II.zip"

    logger.info("downloading %s", SOURCE_URL)
    # UCI rejects the default urllib user agent.
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(zip_path, "wb") as handle:
        handle.write(response.read())
    logger.info("downloaded %.1f MB", zip_path.stat().st_size / 1024**2)

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        logger.info("archive contains: %s", members)
        archive.extractall(RAW_DIR)

    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"expected {RAW_XLSX.name} in the archive, found {members}"
        )

    zip_path.unlink()
    logger.info("extracted to %s (%.1f MB)", RAW_XLSX, RAW_XLSX.stat().st_size / 1024**2)
    return RAW_XLSX


def load_raw() -> pl.DataFrame:
    """Read both sheets of the workbook into one Polars frame.

    Adds `source_sheet` so per-year row counts stay checkable after concatenation.
    Columns are renamed to snake_case; nothing else is altered.
    """
    frames = []
    for sheet in SHEET_NAMES:
        frame = pl.read_excel(RAW_XLSX, sheet_name=sheet, schema_overrides=SOURCE_SCHEMA)

        missing = set(COLUMN_RENAMES) - set(frame.columns)
        if missing:
            raise ValueError(f"sheet {sheet!r} is missing expected columns: {missing}")

        frame = frame.rename(COLUMN_RENAMES).with_columns(
            pl.lit(sheet).alias("source_sheet")
        )
        logger.info("sheet %-16s -> %8d rows", sheet, frame.height)
        frames.append(frame)

    combined = pl.concat(frames, how="vertical")
    logger.info("combined -> %d rows, %d columns", combined.height, combined.width)
    validate_raw(combined)
    return combined


def validate_raw(df: pl.DataFrame) -> None:
    """Guard the acquisition contract. Raises rather than warns.

    The null-invoice assertion is a regression guard: a dtype-inference bug silently
    nulled every cancellation invoice here once, and the failure mode was invisible
    downstream. See SOURCE_SCHEMA.
    """
    expected = {"invoice": pl.String, "stock_code": pl.String, "quantity": pl.Int64}
    for column, dtype in expected.items():
        if df.schema[column] != dtype:
            raise TypeError(f"{column} should be {dtype}, got {df.schema[column]}")

    n_null_invoice = df["invoice"].null_count()
    if n_null_invoice:
        raise ValueError(
            f"{n_null_invoice} rows have a null invoice, which means dtype inference "
            "dropped values. The schema should be pinned, not inferred."
        )

    n_cancellations = df.filter(pl.col("invoice").str.starts_with("C")).height
    if n_cancellations == 0:
        raise ValueError(
            "no C-prefixed cancellation invoices found, which means they were parsed away"
        )

    logger.info(
        "validated: 0 null invoices, %s cancellation rows present",
        f"{n_cancellations:,}",
    )


def build_raw_parquet(force: bool = False) -> pl.DataFrame:
    """Materialise the raw workbook as Parquet and return the frame."""
    if RAW_PARQUET.exists() and not force:
        logger.info("raw parquet already present, loading it")
        return pl.read_parquet(RAW_PARQUET)

    combined = load_raw()
    combined.write_parquet(RAW_PARQUET)
    logger.info(
        "wrote %s (%.1f MB)", RAW_PARQUET.name, RAW_PARQUET.stat().st_size / 1024**2
    )
    return combined


if __name__ == "__main__":
    download_raw()
    build_raw_parquet()
