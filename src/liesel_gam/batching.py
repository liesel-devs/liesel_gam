"""DataFrame helpers for memory-safe mini-batch model setup."""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .category_mapping import series_is_categorical


def category_coverage_indices(
    data: pd.DataFrame, columns: Sequence[str] | None = None
) -> np.ndarray:
    """
    Return rows that cover every observed categorical level.

    The returned positional indices contain the first non-missing occurrence of
    every observed level in every selected column. They can be reserved for the
    training partition before drawing a basis setup sample.

    Parameters
    ----------
    data
        Data whose categorical levels need coverage.
    columns
        Categorical columns to inspect. If ``None``, columns with string, object,
        or pandas categorical dtype are inferred. Pass an empty sequence to disable
        categorical coverage.

    Returns
    -------
    Sorted, unique positional row indices with integer dtype. Missing values do not
    contribute a coverage row.

    Notes
    -----
    Indices are positional, independent of the DataFrame index, and returned in
    ascending order. Missing categorical values are not treated as levels.

    Examples
    --------
    >>> import pandas as pd
    >>> from liesel_gam import category_coverage_indices
    >>> data = pd.DataFrame({"group": ["a", "a", "b", "c"]})
    >>> category_coverage_indices(data, columns=["group"]).tolist()
    [0, 2, 3]

    See Also
    --------
    basis_setup_sample
        Draw a representative setup sample from selected training rows.
    liesel_gam.PandasRegistry.observed_position
        Encode full data using the model's observed variables.
    :doc:`Large-data model setup </notebooks_large_data>`
        Compose the helpers in an executable workflow.
    """
    inferred = columns is None
    if columns is None:
        columns = [name for name in data.columns if series_is_categorical(data[name])]

    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")

    indices: set[int] = set()
    for name in columns:
        series = data[name]
        if not series_is_categorical(series):
            raise TypeError(
                f"Column {name!r} does not have a categorical dtype. Cast numeric "
                "category codes to a pandas categorical dtype first."
            )
        if inferred and len(series) and series.nunique(dropna=True) == len(series):
            warnings.warn(
                f"Inferred categorical column {name!r} has one observed level per "
                "row, so category coverage will reserve every row. Pass columns "
                "explicitly if this column is an identifier.",
                UserWarning,
                stacklevel=2,
            )
        first = series.notna() & ~series.duplicated()
        indices.update(np.flatnonzero(first))

    return np.asarray(sorted(indices), dtype=int)


def basis_setup_sample(
    data: pd.DataFrame,
    *,
    indices: Sequence[int],
    continuous: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    n: int = 2_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Draw a representative sample for setting up GAM bases.

    The sample always includes the minimum and maximum eligible value of every
    continuous column. Remaining rows are selected randomly without replacement.
    Categorical columns retain the training categories even when a category has no
    selected row.

    Parameters
    ----------
    data
        Full data containing the eligible training rows.
    indices
        Positional row indices eligible for the setup sample. They normally come from
        a training split. At least one index is required, and every index must be
        between ``0`` and ``len(data) - 1``.
    continuous
        Columns whose eligible minima and maxima must be retained. If ``None``,
        numeric non-boolean columns are inferred. Pass an empty sequence to disable
        continuous boundary retention.
    categorical
        Columns whose eligible category metadata must be preserved. If ``None``,
        columns with string, object, or pandas categorical dtype are inferred. Pass
        an empty sequence to disable category preservation.
    n
        Target maximum number of sampled rows. Mandatory boundary rows are never
        dropped, so the returned sample can contain more than ``n`` rows.
    seed
        Seed for sampling non-boundary rows. ``None`` uses NumPy's default random
        seeding.

    Returns
    -------
    A copy of the selected rows in source order. Selected categorical columns have
    pandas categorical dtype with metadata derived from the eligible training rows.


    Notes
    -----
    ``n`` is a target rather than a strict limit because continuous boundary rows
    take precedence. The returned rows follow their order in ``data``; categorical
    metadata comes only from the eligible rows.

    Examples
    --------
    >>> import pandas as pd
    >>> from liesel_gam import basis_setup_sample
    >>> data = pd.DataFrame(
    ...     {
    ...         "x": [5.0, 0.0, 10.0, 2.0],
    ...         "group": ["a", "b", "a", "c"],
    ...     }
    ... )
    >>> sample = basis_setup_sample(
    ...     data,
    ...     indices=[0, 1, 2, 3],
    ...     continuous=["x"],
    ...     categorical=["group"],
    ...     n=2,
    ...     seed=1,
    ... )
    >>> sample.to_dict(orient="list")
    {'x': [0.0, 10.0], 'group': ['b', 'a']}
    >>> sample["group"].cat.categories.tolist()
    ['a', 'b', 'c']

    See Also
    --------
    category_coverage_indices
        Find categorical coverage rows to reserve in training.
    liesel_gam.PandasRegistry.observed_position
        Encode full data using the model's observed variables.
    :doc:`Large-data model setup </notebooks_large_data>`
        Compose the helpers in an executable workflow.
    """
    if n < 1:
        raise ValueError("n must be positive.")

    eligible = np.unique(np.asarray(indices, dtype=int))
    if not len(eligible):
        raise ValueError("indices must contain at least one row.")
    if eligible[0] < 0 or eligible[-1] >= len(data):
        raise IndexError("indices contain rows outside the data.")

    if continuous is None:
        continuous = [
            name
            for name in data.columns
            if pd.api.types.is_numeric_dtype(data[name])
            and not pd.api.types.is_bool_dtype(data[name])
        ]
    if categorical is None:
        categorical = [
            name for name in data.columns if series_is_categorical(data[name])
        ]

    missing = sorted((set(continuous) | set(categorical)) - set(data.columns))
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")
    not_categorical = [
        name for name in categorical if not series_is_categorical(data[name])
    ]
    if not_categorical:
        raise TypeError(
            f"Columns {not_categorical} do not have a categorical dtype. Cast "
            "numeric category codes to a pandas categorical dtype first."
        )

    required: set[int] = set()
    for name in continuous:
        values = data.iloc[eligible][name].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Continuous column {name!r} contains non-finite values.")
        required.add(int(eligible[np.argmin(values)]))
        required.add(int(eligible[np.argmax(values)]))

    target = min(n, len(eligible))
    selected = np.asarray(sorted(required), dtype=int)
    if len(selected) > target:
        warnings.warn(
            f"{len(selected)} boundary rows are required, exceeding the requested "
            f"setup sample size n={n}; all boundary rows are retained.",
            UserWarning,
            stacklevel=2,
        )
    remaining_n = max(0, target - len(selected))
    if remaining_n:
        remaining = np.setdiff1d(eligible, selected, assume_unique=True)
        sampled = np.random.default_rng(seed).choice(
            remaining, size=remaining_n, replace=False
        )
        selected = np.concatenate((selected, sampled))

    sample = data.iloc[np.sort(selected)].copy()
    for name in categorical:
        source = data.iloc[eligible][name]
        if isinstance(source.dtype, pd.CategoricalDtype):
            categories = source.cat.categories
            ordered = source.cat.ordered
        else:
            categories = pd.Categorical(source).categories
            ordered = False
        sample[name] = pd.Categorical(
            sample[name], categories=categories, ordered=ordered
        )

    return sample
