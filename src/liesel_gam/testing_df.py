"""
# everything
df_all = make_formula_stress_test_df(n=200, random_state=42)

# minimal set for a quick test
df_small = make_formula_stress_test_df(
    n=50,
    include=("numeric", "categorical", "weird_names"),
    perturb=False,
)
"""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

# -------- Blocks --------------------------------------------------------------


def block_numeric(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x_int": rng.integers(0, 10, size=n),
            "x_float": rng.normal(10, 2, size=n),
            "x_uint8": rng.integers(0, 255, size=n, dtype=np.uint8),
            "x_Int64": (
                pd.Series(rng.integers(0, 5, size=n))
                .mask(rng.random(n) < 0.1)
                .astype("Int64")
            ),
        }
    )


def block_categorical(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cat_unordered": pd.Categorical(rng.choice(list("ABC"), size=n)),
            "cat_ordered": pd.Categorical(
                rng.choice(["low", "med", "high"], size=n),
                categories=["low", "med", "high"],
                ordered=True,
            ),
        }
    )


def block_logical(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flag_bool": rng.choice([True, False, None], size=n).astype("bool"),
        }
    )


def block_strings(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": pd.Series(rng.choice(["α", "β", "gamma"], size=n), dtype="string"),
        }
    )


def block_time(n: int, rng: np.random.Generator) -> pd.DataFrame:
    date0 = pd.to_datetime("2020-01-01")
    return pd.DataFrame(
        {
            "date": date0 + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
            "date_tz": (
                date0 + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
            ).tz_localize("UTC"),
            "td": pd.to_timedelta(rng.integers(-1000, 1000, n), unit="s"),
            "period_m": (
                pd.period_range("2020-01", periods=n, freq="M")
                .to_series()
                .sample(
                    n,
                    replace=True,
                    # high is exclusive; dtype ensures a pandas-friendly 32-bit seed
                    random_state=int(rng.integers(0, 2**32, dtype=np.uint32)),
                )
                .reset_index(drop=True)
            ),
        }
    )


def block_sparse(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x_sparse": pd.arrays.SparseArray((rng.random(n) < 0.05).astype(int)),
        }
    )


def block_weird_names(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(index=range(n))
    df["with space"] = rng.integers(0, 3, size=n)
    df["weird:col*name"] = rng.integers(0, 2, size=n)
    return df


def apply_perturbations(
    df: pd.DataFrame,
    rng: np.random.Generator,
    col: str = "x_float",
    n_nan: int = 5,
    n_inf: int = 3,
) -> pd.DataFrame:
    """Inject NaNs and +inf into a numeric column if it exists."""
    if col in df.columns and len(df) >= (n_nan + n_inf):
        nan_idx = rng.choice(len(df), n_nan, replace=False)
        inf_idx = rng.choice(len(df), n_inf, replace=False)
        df = df.copy()
        df.loc[nan_idx, col] = np.nan
        df.loc[inf_idx, col] = np.inf
    return df


# -------- Orchestrator --------------------------------------------------------

BLOCKS: dict[str, Callable[[int, np.random.Generator], pd.DataFrame]] = {
    "numeric": block_numeric,
    "categorical": block_categorical,
    "logical": block_logical,
    "strings": block_strings,
    "time": block_time,
    "sparse": block_sparse,
    "weird_names": block_weird_names,
}


def make_test_df(
    n: int = 100,
    random_state: int = 0,
    include: Iterable[str] = (
        "numeric",
        "categorical",
        "logical",
        "strings",
        "time",
        "sparse",
        "weird_names",
    ),
    perturb: bool = True,
) -> pd.DataFrame:
    """
    Build a mixed-type DataFrame from modular blocks.

    Parameters
    ----------
    n : number of rows
    random_state : RNG seed
    include : which blocks to include (subset of BLOCKS keys)
    perturb : if True, inject NaNs/+inf into 'x_float'
    """
    rng = np.random.default_rng(random_state)
    frames = []
    for name in include:
        if name not in BLOCKS:
            raise ValueError(f"Unknown block '{name}'. Valid: {tuple(BLOCKS)}")
        frames.append(BLOCKS[name](n, rng))

    df = pd.concat(frames, axis=1)

    if perturb:
        df = apply_perturbations(df, rng, col="x_float", n_nan=5, n_inf=3)

    return df
