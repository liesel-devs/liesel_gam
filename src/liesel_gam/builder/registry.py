"""Variable registry for managing data variables and transformations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd

from .errors import (
    JAXCompatibilityError,
    TypeMismatchError,
    VariableNotFoundError,
    VariableTransformError,
)

Array = Any


class VariableRegistry:
    """Registry for managing variables and their transformations.

    Handles conversion from pandas DataFrame to liesel.Var objects,
    applies transformations, and caches results for efficiency.
    """

    def __init__(
        self, data: pd.DataFrame, na_action: Literal["error", "drop"] = "error"
    ):
        """Initialize the variable registry.

        Args:
            data: pandas DataFrame containing model variables
            na_action: How to handle NaN values. Either "error" or "drop"
        """
        if na_action not in ["error", "drop"]:
            raise ValueError("na_action must be 'error' or 'drop'")

        self.original_data = data.copy()
        self.na_action = na_action
        self.data = self._validate_data(data)
        self._var_cache: dict[str, lsl.Var] = {}

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
            raise JAXCompatibilityError(
                var_name, "could not convert to JAX array"
            ) from e

        return array

    def _get_cache_key(
        self, name: str, transform: Callable | None, var_name: str | None
    ) -> str:
        """Generate cache key for variable with optional transform."""
        if transform is None:
            return name

        transform_id = getattr(transform, "__name__", str(transform))
        cache_name = var_name or f"{name}_{transform_id}"
        return cache_name

    def get_var(
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
            raise VariableNotFoundError(name, available)

        # check if already cached
        if name in self._var_cache:
            var = self._var_cache[name]
        else:
            # get raw values
            values = self._to_jax(self.data[name].to_numpy(), name)
            var = lsl.Var.new_obs(values, name=name)
            self._var_cache[name] = var

        return var

    def _make_transformed_var(
        self, base_var: lsl.Var, transform: Callable, var_name: str | None
    ) -> lsl.Var:
        """Apply a transformation to a base variable and return a new Var."""
        transform_name = (
            f"{base_var.name}_{getattr(transform, '__name__', str(transform))}"
        )

        try:
            transformed_var = lsl.Var.new_calc(
                transform, base_var, name=var_name if var_name else transform_name
            )
        except Exception as e:
            transform_name = getattr(transform, "__name__", str(transform))
            raise VariableTransformError(base_var.name, transform_name, str(e))

        return transformed_var

    def get_transformed_var(
        self,
        name: str,
        transform: Callable,
        var_name: str | None = None,
    ) -> lsl.Var:
        """Get a transformed version of the variable.

        Transformed variables are not cached.

        Args:
            name: Column name in the data
            transform: Callable transformation function to apply
            var_name: Custom name for the resulting variable
        Returns:
            liesel.Var object with transformed values
        """

        base_var = self.get_var(name)
        var = self._make_transformed_var(base_var, transform, var_name)
        return var

    def get_centered_var(self, name: str, var_name: str | None = None) -> lsl.Var:
        """Get a centered version of the variable: x - mean(x).

        note, mean(x) is computed from the original data and cached.

        Args:
            name: Column name in the data
            var_name: Custom name for the resulting variable

        Returns:
            liesel.Var object with centered values
        """
        base_var = self.get_var(name)
        values = base_var.value

        mean_val = float(np.mean(values))

        def center_transform(x):
            return x - mean_val

        center_transform.__name__ = "centered"

        return self._make_transformed_var(
            base_var, center_transform, var_name or f"{name}_centered"
        )

    def get_std_var(self, name: str, var_name: str | None = None) -> lsl.Var:
        """Get a standardized version of the variable: (x - mean(x)) / std(x).

        note, mean(x) and std(x) are computed from the original data and cached.

        Args:
            name: Column name in the data
            var_name: Custom name for the resulting variable

        Returns:
            liesel.Var object with standardized values
        """
        base_var = self.get_var(name)
        values = base_var.value

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        if std_val == 0:
            raise VariableTransformError(
                name,
                "standardization",
                "standard deviation is zero (constant variable)",
            )

        def std_transform(x):
            return (x - mean_val) / std_val

        std_transform.__name__ = "std"

        return self._make_transformed_var(
            base_var, std_transform, var_name or f"{name}_std"
        )

    def get_dummy_vars(self, name: str, var_name_prefix: str | None = None) -> lsl.Var:
        """Get dummy variables for a categorical column using standard dummy coding.

        Args:
            name: Column name in the data
            var_name_prefix: Prefix for dummy variable names

        Returns:
            Dictionary mapping category names to liesel.Var objects
        """

        base_var, codebook = self.get_categorical_var(name)
        base_var.name = base_var.name = f"{name}_codes"

        if len(codebook) < 2:
            raise VariableTransformError(
                name, "dummy encoding", f"only {len(codebook)} unique value(s) found"
            )

        # jax-compatible dummy coding transformation
        n_categories = len(codebook)

        def dummy_transform(codes):
            # create dummy matrix with standard dummy coding (drop first category)
            dummy_matrix = jnp.zeros(
                (codes.shape[0], n_categories - 1), dtype=jnp.int32
            )
            for i in range(1, n_categories):  # only a few cat, so for loop is fine
                dummy_matrix = dummy_matrix.at[:, i - 1].set(codes == i)
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
            raise VariableNotFoundError(name, available)

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
            raise VariableNotFoundError(name, available)

        return isinstance(self.data[name].dtype, pd.CategoricalDtype)

    def is_boolean(self, name: str) -> bool:
        """Check if a variable is boolean.

        Args:
            name: Column name in the data

        Returns:
            True if variable is boolean, False otherwise
        """
        if name not in self.data.columns:
            available = list(self.data.columns)
            raise VariableNotFoundError(name, available)

        return pd.api.types.is_bool_dtype(self.data[name])

    def get_numeric_var(self, name: str) -> lsl.Var:
        """Get a variable and ensure it is numeric.

        Args:
            name: Variable name to retrieve

        Returns:
            liesel.Var object for the numeric variable

        Raises:
            TypeMismatchError: If the variable is not numeric
        """
        if not self.is_numeric(name):
            raise TypeMismatchError(name, "numeric", str(self.data[name].dtype))
        return self.get_var(name)

    def get_categorical_var(self, name: str) -> tuple[lsl.Var, dict[int, Any]]:
        """Get a variable and ensure it is categorical.

        Each variable is converted to integer codes.

        Args:
            name: Variable name to retrieve

        Returns:
            liesel.Var object for the categorical variable and a dictionary
            mapping integer codes to category labels

        Raises:
            TypeMismatchError: If any variable is not categorical
        """
        if not self.is_categorical(name):
            raise TypeMismatchError(name, "categorical", str(self.data[name].dtype))

        # convert categorical variables to integer codes
        values = self.data[name]
        category_codes = values.cat.codes.to_numpy().astype(int)
        category_labels = values.cat.categories.tolist()

        coding_dict = {
            int(code): label for label, code in zip(category_labels, category_codes)
        }

        # check if already cached
        if name in self._var_cache:
            var = self._var_cache[name]
        else:
            jax_codes = self._to_jax(category_codes, name)
            var = lsl.Var.new_obs(jax_codes, name=name)
            self._var_cache[name] = var

        return var, coding_dict

    def get_boolean_var(self, name: str) -> lsl.Var:
        """Get a variable and ensure it is boolean.

        Args:
            name: Variable name to retrieve

        Returns:
            liesel.Var object for the boolean variable

        Raises:
            TypeMismatchError: If the variable is not boolean
        """
        if not self.is_boolean(name):
            raise TypeMismatchError(name, "boolean", str(self.data[name].dtype))
        return self.get_var(name)
