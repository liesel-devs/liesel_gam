"""Load committed mgcv example-data assets without importing R."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def load_columb() -> pd.DataFrame:
    """Load the Columbus data with the dtypes previously produced by ryp."""

    data = pd.read_csv(HERE / "columb.csv", index_col=0, dtype={"district": str})
    levels = [str(index) for index in range(data["district"].nunique())]
    data["district"] = pd.Categorical(data["district"], categories=levels)
    data.index = data.index.astype(str)
    data.index.name = "index"
    return data


def load_columb_polys() -> dict[str, np.ndarray]:
    """Load the Columbus polygons in the legacy liesel-gam coordinate system."""

    data = pd.read_csv(HERE / "columb_polys.csv", dtype={"label": str})
    return {
        str(label): group.sort_values("vertex")[["x", "y"]].to_numpy() - 1.0
        for label, group in data.groupby("label", sort=False)
    }
