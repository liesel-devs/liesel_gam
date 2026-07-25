"""Helpers for loading static data used by the integration tests."""

from pathlib import Path

import pandas as pd

_COLUMB_PATH = Path(__file__).with_name("mgcv_data") / "columb.csv"


def _load_columb() -> pd.DataFrame:
    result = pd.read_csv(_COLUMB_PATH, dtype={"district": object})
    result["district"] = pd.Categorical(
        result["district"],
        categories=[str(index) for index in range(len(result))],
    )
    return result


# Build the categorical dtype at import time, before any optional R/Arrow
# conversion can alter pandas' string inference state.
_COLUMB_DATA = _load_columb()


def columb_to_pandas(*, reset_index: bool = False) -> pd.DataFrame:
    """Load the ``mgcv::columb`` fixture without requiring an R session."""
    result = _COLUMB_DATA.copy(deep=True)
    return result.reset_index() if reset_index else result
