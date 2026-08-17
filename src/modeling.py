"""Shared modeling pipeline pieces.

`build_preprocessor` is the one `ColumnTransformer` every model in this project fits
inside, from the Stage 6 Logistic Regression baseline through the Stage 7 tree models.
Defining it once here is what makes "same preprocessor object type as the baseline"
(the Stage 7 requirement) a fact about the code rather than a convention to remember.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.config import CONTINUOUS_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Scale continuous columns, pass binary columns through unchanged.

    `verbose_feature_names_out=False` keeps clean names like `recency_days` on the
    transformed output instead of `scale__recency_days`, which matters for reading
    coefficients back off a fitted Logistic Regression and for SHAP in Stage 9.
    """
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), list(CONTINUOUS_COLUMNS))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
