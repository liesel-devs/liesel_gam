"""Variable registry for managing data variables and transformations."""

from __future__ import annotations

import hashlib
import inspect
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, assert_never

import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd

from .category_mapping import CategoryMapping, series_is_categorical

logger = logging.getLogger(__name__)

Array = Any


class CannotHashValueError(Exception):
    """Custom exception for values that cannot be hashed."""

    def __init__(self, value: Any):
        super().__init__(f"Cannot hash value of type '{type(value).__name__}'")
        self.value = value


@dataclass
class VarAndMapping:
    var: lsl.Var
    mapping: CategoryMapping | None = None

    @property
    def is_categorical(self) -> bool:
        return self.mapping is not None


class PandasRegistry:
    """Registry for managing variables and their transformations.

    Handles conversion from `pandas.DataFrame` to `liesel.Var` objects,
    applies transformations, and caches results for efficiency.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        na_action: Literal["error", "drop", "ignore"] = "error",
        prefix_names_by: str = "",
    ):
        """Initialize the variable registry.

        Args:
            data: pandas DataFrame containing model variables
            na_action: How to handle NaN values. Either "error", "drop", or "ignore"
        """
        if na_action not in ["error", "drop", "ignore"]:
            raise ValueError("na_action must be 'error', 'drop', or 'ignore'")

        self.original_data = data.copy()
        self.na_action = na_action
        self.data = self._validate_data(data)
        self._var_cache: dict[str, lsl.Var] = {}
        self._derived_cache: dict[str, lsl.Var] = {}
        self.prefix = prefix_names_by

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data and handle NaN values according to policy."""
        if data.isna().any().any():
            if self.na_action == "error":
                na_cols = data.columns[data.isna().any()].tolist()
                raise ValueError(
                    f"Data contains NaN values in columns: {na_cols}. "
                    "Use na_action='drop' to automatically remove rows with NaN values."
                )
            elif self.na_action == "drop":
                clean_data = data.dropna()
                if len(clean_data) == 0:
                    raise ValueError("No rows remaining after dropping NaN values")
                return clean_data
            elif self.na_action == "ignore":
                pass
            else:
                assert_never()

        return data.copy()

    @property
    def columns(self) -> list[str]:
        """Get list of available column names."""
        return list(self.data.columns)

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of the data after NA handling."""
        return self.data.shape

    def _to_jax(self, values: Any, var_name: str) -> Array:
        """Check if values are compatible with JAX."""
        try:
            array = jnp.asarray(values)
        except Exception as e:
            raise TypeError(
                f"Variable '{var_name}' could not convert to JAX array"
            ) from e

        return array

    def _is_closure(self, func: Callable) -> bool:
        """Check if function is a closure (captures variables from outer scope)."""
        return func.__closure__ is not None

    def _hash_closure_value(self, value: Any) -> str:
        """Create hash for closure values, specifically supporting JAX arrays."""
        try:
            # try direct hashing first
            return str(hash(value))
        except TypeError:
            # handle unhashable types
            if isinstance(value, jnp.ndarray):
                # JAX arrays: hash shape, dtype, and content
                return f"jax_array_{value.shape}_{value.dtype}_{hash(value.tobytes())}"
            else:
                # unsupported type - signal to skip caching
                raise CannotHashValueError(value)

    def _hash_function(self, func: Callable) -> str | None:
        """Create hash for function, or use object ID for methods/callable objects."""
        if inspect.isfunction(func):
            # Regular functions: hash source code and closures
            source = inspect.getsource(func)

            if self._is_closure(func):
                # for mypy
                assert func.__closure__ is not None, "Closure should have a closure"
                # hash closure variables
                closure_names = func.__code__.co_freevars
                closure_values = [cell.cell_contents for cell in func.__closure__]

                closure_hashes = []
                for name, value in zip(closure_names, closure_values):
                    try:
                        value_hash = self._hash_closure_value(value)
                        closure_hashes.append(f"{name}:{value_hash}")
                    except CannotHashValueError:
                        # unsupported closure variable, skip caching
                        warnings.warn(
                            f"Function uses unsupported closure variable type "
                            f"'{type(value).__name__}'. Provide explicit cache_key "
                            f"for caching.",
                            UserWarning,
                            stacklevel=3,
                        )
                        return None

                closure_signature = ",".join(sorted(closure_hashes))
            else:
                closure_signature = ""

            # combine source and closure state
            combined = f"{source}|{closure_signature}"
            return hashlib.md5(combined.encode()).hexdigest()

        elif inspect.ismethod(func):
            # Bound method: use object ID + method name for consistent caching
            obj_id = id(func.__self__)
            method_name = func.__name__
            return f"method_{obj_id}_{method_name}"

        elif hasattr(func, "__call__"):
            # Callable objects, lambdas, etc.: use object ID
            return f"obj_id_{id(func)}"
        else:
            raise TypeError(f"Unsupported function type: {type(func)}")

    def get_obs(
        self,
        name: str,
    ) -> lsl.Var:
        """Get or create a liesel Var for a data column.

        Args:
            name: Column name in the data

        Returns:
            liesel.Var object
        """
        if name not in self.data.columns:
            available = list(self.data.columns)
            raise KeyError(
                f"Variable '{name}' not found in data. "
                f"Available variables: {sorted(available)}"
            )

        varname = self.prefix + name

        # check if already cached
        if name in self._var_cache:
            var = self._var_cache[name]
        else:
            # get raw values
            values = self._to_jax(self.data[name].to_numpy(), name)
            var = lsl.Var.new_obs(values, name=varname)
            self._var_cache[name] = var

        return var

    def _make_derived_var(
        self, base_var: lsl.Var, transform: Callable, var_name: str | None
    ) -> lsl.Var:
        """Apply a transformation to a base variable and return a new Var."""
        if var_name is None:
            var_name = (
                f"{base_var.name}_{getattr(transform, '__name__', str(transform))}"
            )

        try:
            derived_var = lsl.Var.new_calc(transform, base_var, name=var_name)
        except Exception as e:
            transformation_name = getattr(transform, "__name__", str(transform))
            raise ValueError(
                f"Failed to apply transformation '{transformation_name}' "
                f"to variable '{base_var.name}': {str(e)}"
            )

        return derived_var

    def get_calc(
        self,
        name: str,
        transform: Callable,
        var_name: str | None = None,
        cache_key: str | None = None,
    ) -> lsl.Var:
        """Get a derived version of the variable.

        Derived variables are cached when possible. Creates a lsl.new_obs for the
        base variable and a lsl.new_calc for the derived variable.

        Args:
            name: Column name in the data frame
            transform: Callable transformation function to apply
            var_name: Custom name for the resulting variable
            cache_key: Explicit cache key. If provided, skips function hashing.
        Returns:
            liesel.Var object with transformed values
        """

        # get base var
        base_var = self.get_obs(name)

        # generate cache key
        if cache_key is not None:
            # explicit cache key provided
            full_cache_key = f"{name}_{cache_key}_{var_name or 'default'}"
        else:
            # try to hash the function
            func_hash = self._hash_function(transform)
            if func_hash is None:
                # caching not possible, return derived var without caching
                return self._make_derived_var(base_var, transform, var_name)

            full_cache_key = f"{name}_{func_hash}_{var_name or 'default'}"

        # check cache first
        if full_cache_key in self._derived_cache:
            return self._derived_cache[full_cache_key]

        # cache miss
        var = self._make_derived_var(base_var, transform, var_name)
        self._derived_cache[full_cache_key] = var

        return var

    def get_calc_centered(self, name: str, var_name: str | None = None) -> lsl.Var:
        """Get a centered version of the variable: x - mean(x).

        note, mean(x) is computed from the original data and cached.

        Args:
            name: Column name in the data
            var_name: Custom name for the resulting variable

        Returns:
            liesel.Var object with centered values
        """
        base_var = self.get_obs(name)
        values = base_var.value

        mean_val = float(np.mean(values))

        def center_transform(x):
            return x - mean_val

        center_transform.__name__ = "centered"

        return self._make_derived_var(
            base_var, center_transform, var_name or f"{name}_centered"
        )

    def get_calc_standardized(self, name: str, var_name: str | None = None) -> lsl.Var:
        """Get a standardized version of the variable: (x - mean(x)) / std(x).

        note, mean(x) and std(x) are computed from the original data and cached.

        Args:
            name: Column name in the data
            var_name: Custom name for the resulting variable

        Returns:
            liesel.Var object with standardized values
        """
        base_var = self.get_obs(name)
        values = base_var.value

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        if std_val == 0:
            raise ValueError(
                f"Failed to apply transformation 'standardization' to variable "
                f"'{name}': standard deviation is zero (constant variable)"
            )

        def std_transform(x):
            return (x - mean_val) / std_val

        std_transform.__name__ = "std"

        return self._make_derived_var(
            base_var, std_transform, var_name or f"{name}_std"
        )

    def get_calc_dummymatrix(
        self, name: str, var_name_prefix: str | None = None
    ) -> lsl.Var:
        """Get dummy matrix for a categorical column using standard dummy coding.

        Drops the column of the first category.

        Args:
            name: Column name in the data
            var_name_prefix: Prefix for dummy variable names

        Returns:
            Dictionary mapping category names to liesel.Var objects
        """

        base_var, mapping = self.get_categorical_obs(name)
        base_var.name = base_var.name = f"{name}_codes"

        codebook = mapping.labels_to_integers_map

        if len(codebook) < 2:
            raise ValueError(
                f"Failed to apply transformation 'dummy encoding' to variable "
                f"'{name}': only {len(codebook)} unique value(s) found"
            )

        # jax-compatible dummy coding transformation
        n_categories = len(codebook)

        def dummy_transform(codes):
            # create dummy matrix with standard dummy coding (drop first category)
            # use float32 to support NaN for unknown codes
            dummy_matrix = jnp.zeros(
                (codes.shape[0], n_categories - 1), dtype=jnp.float32
            )
            for i in range(1, n_categories):  # only a few cat, so for loop is fine
                dummy_matrix = dummy_matrix.at[:, i - 1].set(codes == i)

            # set rows with unknown codes (>= n_categories or < 0) to NaN
            unknown_mask = (codes >= n_categories) | (codes < 0)
            dummy_matrix = jnp.where(unknown_mask[:, None], jnp.nan, dummy_matrix)

            return dummy_matrix

        dummy_transform.__name__ = f"{name}_dummy"

        # create dummy matrix variable
        prefix = var_name_prefix or f"{name}_"
        dummy_matrix_name = f"{prefix}matrix"
        dummy_matrix_var = lsl.Var.new_calc(
            dummy_transform, base_var, name=dummy_matrix_name
        )

        return dummy_matrix_var

    def is_numeric(self, name: str) -> bool:
        """Check if a variable is numeric.

        Args:
            name: Column name in the data

        Returns:
            True if variable is numeric, False otherwise
        """
        if name not in self.data.columns:
            available = list(self.data.columns)
            raise KeyError(
                f"Variable '{name}' not found in data. "
                f"Available variables: {sorted(available)}"
            )

        return pd.api.types.is_numeric_dtype(self.data[name])

    def is_categorical(self, name: str) -> bool:
        """Check if a variable is categorical.

        Args:
            name: Column name in the data

        Returns:
            True if variable is categorical, False otherwise
        """
        if name not in self.data.columns:
            available = list(self.data.columns)
            raise KeyError(
                f"Variable '{name}' not found in data. "
                f"Available variables: {sorted(available)}"
            )

        return series_is_categorical(self.data[name])

    def is_boolean(self, name: str) -> bool:
        """Check if a variable is boolean.

        Args:
            name: Column name in the data

        Returns:
            True if variable is boolean, False otherwise
        """
        if name not in self.data.columns:
            available = list(self.data.columns)
            raise KeyError(
                f"Variable '{name}' not found in data. "
                f"Available variables: {sorted(available)}"
            )

        return pd.api.types.is_bool_dtype(self.data[name])

    def get_numeric_obs(self, name: str) -> lsl.Var:
        """Get a variable and ensure it is numeric.

        Args:
            name: Variable name to retrieve

        Returns:
            liesel.Var object for the numeric variable

        Raises:
            TypeError: If the variable is not numeric
        """
        if not self.is_numeric(name):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected numeric, "
                f"got {str(self.data[name].dtype)}"
            )
        return self.get_obs(name)

    def get_categorical_obs(self, name: str) -> tuple[lsl.Var, CategoryMapping]:
        """Get a variable and ensure it is categorical.

        Each variable is converted to integer codes.

        Args:
            name: Variable name to retrieve

        Returns:
            liesel.Var object for the categorical variable and a CategoryMapping.

        Raises:
            TypeError: If any variable is not categorical
        """
        series = self.data[name]
        if not self.is_categorical(name):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected categorical, "
                f"got {str(series.dtype)}"
            )

        mapping = CategoryMapping.from_series(series)
        if name in self._var_cache:
            var = self._var_cache[name]
        else:
            # convert categorical variables to integer codes
            category_codes = mapping.labels_to_integers(series)
            jax_codes = self._to_jax(category_codes, name)
            varname = self.prefix + name
            var = lsl.Var.new_obs(jax_codes, name=varname)
            self._var_cache[name] = var

            # now some exception handling
            # only emitted once
            nparams = len(mapping.labels_to_integers_map)
            n_observed_clusters = jnp.unique(var.value).size
            observed_clusters = np.unique(var.value).tolist()
            clusters = list(mapping.integers_to_labels_map)
            clusters_not_in_data = [c for c in clusters if c not in observed_clusters]

            if n_observed_clusters != nparams:
                logger.info(
                    f"For {name}, there are {nparams} categories, but the "
                    f"data contain observations for only {n_observed_clusters}. The "
                    f"categories without observations are: {clusters_not_in_data}. "
                    "If this is intended, you can ignore this warning. "
                    "Be aware, that parameters for the unobserved categories may be "
                    "included in the model."
                )

        return var, mapping

    def get_boolean_obs(self, name: str) -> lsl.Var:
        """Get a variable and ensure it is boolean.

        Args:
            name: Variable name to retrieve

        Returns:
            liesel.Var object for the boolean variable

        Raises:
            TypeError: If the variable is not boolean
        """
        if not self.is_boolean(name):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected boolean, "
                f"got {str(self.data[name].dtype)}"
            )
        return self.get_obs(name)

    def get_obs_and_mapping(self, name: str) -> VarAndMapping:
        """
        Get an observed variable. Returns a wrapper that holds the variable and,
        if the variable is categorical, the :class:`.CategoryMapping` between
        labels and integer codes.
        """
        if self.is_categorical(name):
            var, mapping = self.get_categorical_obs(name)
        else:
            var = self.get_obs(name)
            mapping = None

        return VarAndMapping(var, mapping)
