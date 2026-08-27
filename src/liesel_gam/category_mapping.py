from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from types import MappingProxyType
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
    """An immutable mapping between category labels and contiguous integer codes.

    Parameters
    ----------
    labels_to_integers_map
        The category mapping. Codes must be unique and contiguous from zero.
    unknown_category
        Optional catch-all label. If absent from the mapping, it is appended. Unknown
        nonmissing labels are encoded as this category. The default ``None`` rejects
        unknown labels.

    Examples
    --------
    >>> mapping = CategoryMapping.from_labels(["pear", "apple", "pear"])
    >>> dict(mapping.labels_to_codes_map)
    {'apple': 0, 'pear': 1}
    """

    def __init__(
        self,
        labels_to_integers_map: dict[Any, int],
        unknown_category: Any | None = None,
    ) -> None:
        if unknown_category is not None:
            try:
                hash(unknown_category)
            except TypeError:
                raise TypeError("The unknown category must be hashable.") from None
            if bool(pd.isna(unknown_category)):
                raise ValueError("The unknown category must not be missing.")

        mapping = dict(labels_to_integers_map)
        if unknown_category is not None and unknown_category not in mapping:
            mapping[unknown_category] = len(mapping)

        codes = list(mapping.values())
        if not all(isinstance(code, Integral) for code in codes) or set(codes) != set(
            range(len(codes))
        ):
            raise ValueError("Category codes must be unique and contiguous from zero.")

        self._code_for_unknown_label = (
            mapping[unknown_category] if unknown_category is not None else None
        )
        self._label_for_unknown_code = None

        self._labels_to_integers_map = mapping
        self._integers_to_labels_map = {
            code: label for label, code in self._labels_to_integers_map.items()
        }

    @property
    def labels_to_codes_map(self) -> Mapping[Any, int]:
        """The immutable label-to-code mapping."""
        return MappingProxyType(self._labels_to_integers_map)

    @property
    def codes_to_labels_map(self) -> Mapping[int, Any]:
        """The immutable code-to-label mapping."""
        return MappingProxyType(self._integers_to_labels_map)

    @property
    def labels_to_integers_map(self) -> Mapping[Any, int]:
        """Compatibility alias for :attr:`labels_to_codes_map`."""
        return self.labels_to_codes_map

    @property
    def integers_to_labels_map(self) -> Mapping[int, Any]:
        """Compatibility alias for :attr:`codes_to_labels_map`."""
        return self.codes_to_labels_map

    @classmethod
    def from_labels(
        cls,
        values: Any,
        categories: Sequence[Any] | None = None,
        *,
        unknown_category: Any | None = None,
    ) -> CategoryMapping:
        """Create a mapping from labels and optional ordered categories.

        Inferred categories are sorted. Explicit categories preserve their order and
        may include levels absent from ``values``. When ``unknown_category`` is set,
        labels omitted from explicit ``categories`` are mapped to it. Without it,
        such omissions are rejected. Values may have any nonempty rectangular shape;
        one mapping is shared by all entries.

        Parameters
        ----------
        values
            A nonempty rectangular array of hashable, nonmissing labels.
        categories
            Optional ordered categories. They may include unobserved categories.
        unknown_category
            Optional catch-all label for unknown nonmissing labels.
        """
        values_array = np.asarray(values, dtype=object)
        if values_array.size == 0:
            raise ValueError("Categorical labels must be nonempty.")
        values_flat = values_array.reshape(-1).tolist()
        try:
            for label in values_flat:
                hash(label)
        except TypeError:
            raise TypeError("Categorical labels must be hashable.") from None
        if any(bool(pd.isna(label)) for label in values_flat):
            raise ValueError("Categorical labels must not contain missing values.")

        if categories is None:
            if isinstance(values, pd.Categorical):
                categories = values.categories
            elif isinstance(values, pd.Series) and isinstance(
                values.dtype, pd.CategoricalDtype
            ):
                categories = values.cat.categories
            else:
                unique_labels = set(values_flat)
                try:
                    categories = sorted(unique_labels)
                except TypeError:
                    raise TypeError(
                        "Could not sort categorical labels; pass categories explicitly."
                    ) from None

        category_list = list(categories)
        if any(bool(pd.isna(label)) for label in category_list):
            raise ValueError("Categories must not contain missing values.")
        try:
            unique_categories = set(category_list)
        except TypeError:
            raise TypeError("Categories must be hashable.") from None
        if len(unique_categories) != len(category_list):
            raise ValueError("Categories must be unique.")

        mapping = {label: code for code, label in enumerate(category_list)}
        missing = [label for label in values_flat if label not in mapping]
        if missing and unknown_category is None:
            raise ValueError(f"categories omit observed labels: {missing!r}")
        return cls(mapping, unknown_category=unknown_category)

    @classmethod
    def from_series(
        cls,
        series: pd.Series | pd.Categorical,
        *,
        unknown_category: Any | None = None,
    ) -> CategoryMapping:
        """Create a mapping from a pandas Series or categorical.

        When series is a pd.Categorical, the category sorting is kept.
        When series is a series of dtype str or object, categories are sorted
        alphabetically. ``unknown_category`` optionally adds a catch-all category;
        the default rejects unknown labels.

        Parameters
        ----------
        series
            The pandas Series or categorical containing category labels.
        unknown_category
            Optional catch-all label for unknown nonmissing labels.
        """
        if not isinstance(series, pd.Series | pd.Categorical):
            raise TypeError(
                f"series must be a pd.Series or pd.Categorical, got {type(series)}."
            )
        return cls.from_labels(series, unknown_category=unknown_category)

    def to_codes(
        self, labels_or_codes: np.typing.ArrayLike | Sequence[int] | Sequence[str]
    ) -> np.typing.NDArray[np.int_]:
        """Convert labels to codes, passing already encoded integers through.

        Parameters
        ----------
        labels_or_codes
            Category labels or already encoded integer codes.

        .. warning::

            Integer inputs are interpreted as codes, not semantic integer labels. This
            makes the conversion idempotent. Use :meth:`labels_to_codes` when you
            explicitly need to encode integer-valued labels. Invalid integer codes are
            rejected even when a catch-all category is configured.
        """
        arr = np.asarray(labels_or_codes)

        # Case 1: Already an integer array
        if np.issubdtype(arr.dtype, np.integer):
            valid_integers = np.array(list(self.codes_to_labels_map))
            if not np.isin(arr, valid_integers).all():
                invalid = arr[~np.isin(arr, valid_integers)]
                raise ValueError(
                    f"Unknown integer codes: {invalid.tolist()} "
                    f"(valid integers: {valid_integers.tolist()})"
                )
            return arr.astype(int, copy=False)

        # Case 2: Otherwise treat as labels
        return self.labels_to_codes(labels_or_codes)

    def to_integers(
        self, labels_or_integers: np.typing.ArrayLike | Sequence[int] | Sequence[str]
    ) -> np.typing.NDArray[np.int_]:
        """Compatibility alias for :meth:`to_codes`.

        Parameters
        ----------
        labels_or_integers
            Category labels or already encoded integer codes.
        """
        return self.to_codes(labels_or_integers)

    def to_labels(
        self, labels_or_integers: np.typing.ArrayLike | Sequence[int] | Sequence[str]
    ) -> np.typing.NDArray[Any]:
        """Convert codes to labels, passing labels through.

        Parameters
        ----------
        labels_or_integers
            Category labels or integer codes.
        """
        arr = np.asarray(labels_or_integers)

        # Case 1: It is an integer array
        if np.issubdtype(arr.dtype, np.integer):
            return self.codes_to_labels(arr)

        # Case 2: Otherwise treat as labels
        valid_labels = np.array(list(self.labels_to_codes_map))
        if not np.isin(arr, valid_labels).all():
            invalid = arr[~np.isin(arr, valid_labels)]
            raise ValueError(
                f"Unknown labels: {invalid.tolist()} "
                f"(valid labels: {valid_labels.tolist()})"
            )
        return arr

    def labels_to_codes(
        self, labels: np.typing.ArrayLike | Sequence[Any]
    ) -> np.typing.NDArray[np.int_]:
        """Convert category labels to integer codes.

        Unknown nonmissing labels map to the configured catch-all category. If no
        catch-all is configured, they raise :class:`UnknownLabelError`. Missing labels
        always raise.

        Parameters
        ----------
        labels
            Category labels to encode.
        """
        labels = np.asarray(labels, dtype=object)
        labels_flat = labels.flatten()
        codes_flat = np.zeros_like(labels_flat, dtype=int)

        for i, xi in enumerate(labels_flat):
            if bool(pd.isna(xi)):
                raise UnknownLabelError(f"Category label {xi} is unknown.")
            code = self.labels_to_codes_map.get(xi)
            if code is None and self._code_for_unknown_label is None:
                raise UnknownLabelError(f"Category label {xi} is unknown.")
            codes_flat[i] = self._code_for_unknown_label if code is None else code

        codes = np.reshape(codes_flat, shape=labels.shape)

        return np.astype(codes, np.int_)

    def labels_to_integers(
        self, labels: np.typing.ArrayLike | Sequence[Any]
    ) -> np.typing.NDArray[np.int_]:
        """Compatibility alias for :meth:`labels_to_codes`.

        Parameters
        ----------
        labels
            Category labels to encode.
        """
        return self.labels_to_codes(labels)

    def codes_to_labels(
        self, codes: np.typing.ArrayLike | Sequence[int]
    ) -> np.typing.NDArray[Any]:
        """Convert integer codes to category labels.

        Parameters
        ----------
        codes
            Integer codes to decode.
        """
        codes = np.asarray(codes)
        codes_flat = codes.flatten()
        labels_flat_list = []

        for xi in codes_flat:
            label = self.codes_to_labels_map.get(xi, self._label_for_unknown_code)
            if label == self._label_for_unknown_code:
                raise UnknownCodeError(f"Category code {xi} is unknown.")
            labels_flat_list.append(label)

        labels_flat = np.asarray(labels_flat_list, dtype=object)
        labels = np.reshape(labels_flat, shape=codes.shape)
        return labels

    def integers_to_labels(
        self, integers: np.typing.ArrayLike | Sequence[int]
    ) -> np.typing.NDArray[Any]:
        """Compatibility alias for :meth:`codes_to_labels`.

        Parameters
        ----------
        integers
            Integer codes to decode.
        """
        return self.codes_to_labels(integers)


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
    if series.dtype == "string" and series.dtype.name == "string":
        raise TypeError(
            f"Pandas dtype {series.dtype} cannot be safely interpreted as "
            "categorical, please process to dtype str or object."
        )
    return is_cat1 or is_cat2
