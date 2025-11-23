from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

Array = Any


class CategoryError(KeyError):
    pass


class UnknownLabelError(CategoryError):
    pass


class UnknownCodeError(CategoryError):
    pass


class CategoryMapping:
    """Wraps a category mapping of labels to integers."""

    def __init__(self, labels_to_integers_map: dict[Any, int]) -> None:
        self._code_for_unknown_label = -1
        self._label_for_unknown_code = None

        self.labels_to_integers_map = labels_to_integers_map
        self.integers_to_labels_map = {
            code: label for label, code in self.labels_to_integers_map.items()
        }

    @classmethod
    def from_series(cls, series: pd.Series | pd.Categorical) -> CategoryMapping:
        """
        When series is a pd.Categorical, the category sorting is kept.
        When series is a series of dtype str or object, categories are sorted
        alphabetically.
        """
        is_series = isinstance(series, pd.Series)
        has_cat_dtype = isinstance(series.dtype, pd.CategoricalDtype)
        is_cat = isinstance(series, pd.Categorical)
        if is_cat:
            unique_labels = np.asarray(series.categories)
        elif is_series and has_cat_dtype:
            unique_labels = np.asarray(series.cat.categories)
        elif is_series:
            cat = pd.Categorical(series)
            unique_labels = np.sort(np.asarray(cat.categories))
        else:
            raise TypeError(
                f"series must be a pd.Series or pd.Categorical, got {type(series)}."
            )

        mapping = {val: i for i, val in enumerate(unique_labels)}
        return cls(mapping)

    def labels_to_integers(
        self, labels: np.typing.ArrayLike | Sequence[str]
    ) -> np.typing.NDArray[np.int_]:
        """
        A function of labels -> integers.

        For unknown labels, returns -1.
        """
        labels = np.asarray(labels)
        labels_flat = labels.flatten()
        codes_flat = np.zeros_like(labels_flat, dtype=int)

        for i, xi in enumerate(labels_flat):
            codes_flat[i] = self.labels_to_integers_map.get(
                xi, self._code_for_unknown_label
            )
            if codes_flat[i] == self._code_for_unknown_label:
                raise UnknownLabelError(f"Category label {xi} is unknown.")

        codes = np.reshape(codes_flat, shape=labels.shape)

        return np.astype(codes, np.int_)

    def integers_to_labels(
        self, integers: np.typing.ArrayLike | Sequence[int]
    ) -> np.typing.NDArray[Any]:
        """
        A function of integers -> labels.

        For integers without labels, returns
        """
        integers = np.asarray(integers)
        integers_flat = integers.flatten()
        labels_flat_list = []

        for xi in integers_flat:
            label = self.integers_to_labels_map.get(xi, self._label_for_unknown_code)
            if label == self._label_for_unknown_code:
                raise UnknownCodeError(f"Category code {xi} is unknown.")
            labels_flat_list.append(label)

        labels_flat = np.asarray(labels_flat_list)
        labels = np.reshape(labels_flat, shape=integers.shape)
        return labels


def series_is_categorical(series: pd.Series | pd.Categorical) -> bool:
    """
    Provides a liberal interpretation of when a series is categorical. The following
    are treated as categorical:

    - Series with dtype str
    - Series with dtype object
    - Series with dtype CategoricalDtype
    """
    # This corresponds to how formulaic determines categorical columns.
    # See formulaic.materializers.pandas.PandasMaterializer._is_categorical
    is_cat1 = series.dtype in ("str", "object")
    is_cat2 = isinstance(series.dtype, pd.CategoricalDtype)
    return is_cat1 or is_cat2
