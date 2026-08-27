"""Variable registries for constructing Liesel variables from named values."""

from __future__ import annotations

import hashlib
import inspect
import logging
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, assert_never

import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
from liesel.goose.types import Position

from .category_mapping import CategoryMapping, series_is_categorical
from .var import CatVar

logger = logging.getLogger(__name__)

Converter = Callable[[Any], Any] | Literal["default"]


class CannotHashValueError(Exception):
    """A closure value cannot be hashed for the derived-variable cache."""

    def __init__(self, value: Any) -> None:
        super().__init__(f"Cannot hash value of type '{type(value).__name__}'")
        self.value = value


@dataclass
class VarAndMapping:
    """An observed variable and its optional categorical mapping."""

    var: lsl.Var
    mapping: CategoryMapping | None = None

    @property
    def is_categorical(self) -> bool:
        return self.mapping is not None


class DictRegistry:
    """Registry for constructing Liesel variables from a mapping.

    The registry makes a shallow copy of the mapping. Values may have unrelated
    shapes. Mutating :attr:`data` later affects uncached keys, while already-created
    variables remain authoritative. Nested mappings receive no special alignment.

    Parameters
    ----------
    data
        Mapping from string names to source values.
    prefix_names_by
        Prefix for generated Liesel variable names.
    convert
        Default Liesel value converter. The default ``"default"`` preserves Liesel's
        native conversion.

    .. warning::

        A custom converter may perform host-only work for raw values. It may be called
        repeatedly, so it must be idempotent on converted values. If its variable
        participates in compiled state updates, the converter also needs a JAX-safe
        path for already-converted arrays or tracers. :class:`.CatVar` demonstrates
        this dual-path behavior.

    Examples
    --------
    >>> from liesel_gam import DictRegistry
    >>> registry = DictRegistry({"x": [1.0, 2.0]})
    >>> registry.get_obs("x").value
    Array([1., 2.], dtype=float32)
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        prefix_names_by: str = "",
        convert: Converter = "default",
    ) -> None:
        if not all(isinstance(key, str) for key in data):
            raise TypeError("Registry keys must be strings.")
        # Liesel variables and nodes may be supported as source values in the future.
        if any(isinstance(value, lsl.Var | lsl.Node) for value in data.values()):
            raise TypeError(
                "Registry source values must not be Liesel variables or nodes."
            )

        self.data = dict(data)
        self.prefix = prefix_names_by
        self.convert = convert
        self._var_cache: dict[str, lsl.Var] = {}
        self._var_converters: dict[str, Converter] = {}
        self._matrix_cache: dict[
            tuple[tuple[str, ...], bool], lsl.Calc | lsl.TransientCalc
        ] = {}
        self._derived_cache: dict[tuple[str, str, str | None, bool], lsl.Var] = {}

    def keys(self) -> Iterable[str]:
        """Return the current source-value names."""
        return self.data.keys()

    def _require_name(self, name: str) -> None:
        if name not in self.data:
            raise KeyError(
                f"Variable '{name}' not found in data. "
                f"Available variables: {sorted(self.keys())}"
            )

    def _source_value(self, name: str) -> Any:
        self._require_name(name)
        value = self.data[name]
        if isinstance(value, lsl.Var | lsl.Node):
            raise TypeError(
                "Registry source values must not be Liesel variables or nodes."
            )
        return value

    @staticmethod
    def _same_converter(left: Converter, right: Converter) -> bool:
        if isinstance(left, str) or isinstance(right, str):
            return isinstance(left, str) and isinstance(right, str)
        return left is right

    @staticmethod
    def _is_default_converter(converter: Converter) -> bool:
        return isinstance(converter, str) and converter == "default"

    @staticmethod
    def _is_closure(func: Callable) -> bool:
        return inspect.isfunction(func) and func.__closure__ is not None

    @staticmethod
    def _hash_closure_value(value: Any) -> str:
        try:
            return str(hash(value))
        except TypeError:
            if isinstance(value, jnp.ndarray):
                return f"jax_array_{value.shape}_{value.dtype}_{hash(value.tobytes())}"
            raise CannotHashValueError(value) from None

    def _hash_function(self, func: Callable) -> str | None:
        if inspect.isfunction(func):
            source = inspect.getsource(func)
            closure_hashes = []
            if self._is_closure(func):
                assert func.__closure__ is not None
                for name, cell in zip(func.__code__.co_freevars, func.__closure__):
                    try:
                        value_hash = self._hash_closure_value(cell.cell_contents)
                    except CannotHashValueError:
                        warnings.warn(
                            "Function uses unsupported closure variable type "
                            f"'{type(cell.cell_contents).__name__}'. Provide explicit "
                            "cache_key for caching.",
                            UserWarning,
                            stacklevel=3,
                        )
                        return None
                    closure_hashes.append(f"{name}:{value_hash}")
            combined = f"{source}|{','.join(sorted(closure_hashes))}"
            return hashlib.md5(combined.encode()).hexdigest()
        if inspect.ismethod(func):
            return f"method_{id(func.__self__)}_{func.__name__}"
        if callable(func):
            return f"obj_id_{id(func)}"
        raise TypeError(f"Unsupported function type: {type(func)}")

    @staticmethod
    def _make_derived_var(
        base_var: lsl.Var,
        transform: Callable,
        var_name: str | None,
        *,
        cache: bool,
    ) -> lsl.Var:
        if var_name is None:
            var_name = (
                f"{base_var.name}_{getattr(transform, '__name__', str(transform))}"
            )
        try:
            return lsl.Var.new_calc(transform, base_var, name=var_name, cache=cache)
        except Exception as e:
            transformation_name = getattr(transform, "__name__", str(transform))
            raise ValueError(
                f"Failed to apply transformation '{transformation_name}' "
                f"to variable '{base_var.name}': {e!s}"
            ) from e

    def get_obs(
        self,
        name: str,
        *,
        convert: Converter | None = None,
    ) -> lsl.Var:
        """Get or create the canonical observed variable for a source value.

        ``convert=None`` inherits the registry converter. Passing ``"default"`` or a
        callable is an explicit choice. Once a key has been materialized, an explicit
        different converter raises instead of silently returning an incompatible Var.
        Callable identity determines whether two custom converters are the same.
        """
        self._require_name(name)
        if name in self._var_cache:
            if convert is not None and not self._same_converter(
                convert, self._var_converters[name]
            ):
                raise ValueError(
                    f"Variable {name!r} was already created with a different converter."
                )
            return self._var_cache[name]

        converter = self.convert if convert is None else convert
        try:
            var = lsl.Var.new_obs(
                self._source_value(name),
                name=self.prefix + name,
                convert=converter,
            )
        except Exception as e:
            raise TypeError(f"Variable '{name}' could not be converted") from e
        self._var_cache[name] = var
        self._var_converters[name] = converter
        return var

    def is_numeric(self, name: str) -> bool:
        """Whether the canonical converted value has a numeric dtype."""
        if self.is_categorical(name) and (
            isinstance(self._var_cache.get(name), CatVar)
            or name not in self._var_cache
            and self._is_default_converter(self.convert)
        ):
            return False
        return pd.api.types.is_numeric_dtype(np.asarray(self.get_obs(name).value).dtype)

    def is_boolean(self, name: str) -> bool:
        """Whether the canonical converted value has a boolean dtype."""
        if self.is_categorical(name) and (
            isinstance(self._var_cache.get(name), CatVar)
            or name not in self._var_cache
            and self._is_default_converter(self.convert)
        ):
            return False
        return pd.api.types.is_bool_dtype(np.asarray(self.get_obs(name).value).dtype)

    def is_categorical(self, name: str) -> bool:
        """Whether the source uses a supported categorical representation.

        Numeric arrays are deliberately treated as numeric. Integer category labels
        are rejected even in :class:`pandas.Categorical`; convert semantic integer
        labels to strings first. String and object arrays may have any shape.
        """
        value = self._source_value(name)
        if isinstance(value, pd.Categorical):
            return True
        if isinstance(value, pd.Series):
            return series_is_categorical(value)
        array = np.asarray(value)
        return array.dtype.kind in "OUS"

    def get_numeric_obs(
        self,
        name: str,
        *,
        convert: Converter | None = None,
    ) -> lsl.Var:
        """Get an observed variable and require a numeric converted value."""
        converter = self.convert if convert is None else convert
        if self.is_categorical(name) and self._is_default_converter(converter):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected numeric, "
                f"got {np.asarray(self._source_value(name)).dtype!s}"
            )
        var = self.get_obs(name, convert=convert)
        if not pd.api.types.is_numeric_dtype(np.asarray(var.value).dtype):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected numeric, "
                f"got {np.asarray(var.value).dtype!s}"
            )
        return var

    def get_boolean_obs(
        self,
        name: str,
        *,
        convert: Converter | None = None,
    ) -> lsl.Var:
        """Get an observed variable and require a boolean converted value."""
        converter = self.convert if convert is None else convert
        if self.is_categorical(name) and self._is_default_converter(converter):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected boolean, "
                f"got {np.asarray(self._source_value(name)).dtype!s}"
            )
        var = self.get_obs(name, convert=convert)
        if not pd.api.types.is_bool_dtype(np.asarray(var.value).dtype):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected boolean, "
                f"got {np.asarray(var.value).dtype!s}"
            )
        return var

    def get_categorical_obs(self, name: str) -> tuple[CatVar, CategoryMapping]:
        """Get a categorical observed variable and its category mapping.

        Registry inference intentionally rejects semantic integer labels; convert
        them to strings first. Use :meth:`.CatVar.from_codes` only for already encoded
        integer codes.
        """
        source = self._source_value(name)
        if not self.is_categorical(name):
            raise TypeError(
                f"Type mismatch for variable '{name}': expected categorical, "
                f"got {np.asarray(source).dtype!s}"
            )

        if name in self._var_cache:
            var = self._var_cache[name]
            if not isinstance(var, CatVar):
                raise TypeError(f"Cached variable {name!r} is not a CatVar.")
            return var, var.mapping

        var = CatVar(source, name=self.prefix + name)
        self._var_cache[name] = var
        self._var_converters[name] = var._convert
        self._log_unobserved_categories(name, var)
        return var, var.mapping

    @staticmethod
    def _log_unobserved_categories(name: str, var: CatVar) -> None:
        n_categories = len(var.mapping.labels_to_integers_map)
        observed = np.unique(var.value).tolist()
        missing = [code for code in range(n_categories) if code not in observed]
        if missing:
            logger.info(
                "For %s, there are %s categories, but the data contain observations "
                "for only %s. The categories without observations are: %s. If this "
                "is intended, you can ignore this warning. Be aware, that parameters "
                "for the unobserved categories may be included in the model.",
                name,
                n_categories,
                len(observed),
                missing,
            )

    def get_obs_and_mapping(self, name: str) -> VarAndMapping:
        """Get an observed variable and any categorical mapping."""
        if self.is_categorical(name):
            var, mapping = self.get_categorical_obs(name)
            return VarAndMapping(var, mapping)
        return VarAndMapping(self.get_obs(name))

    def get_many_numeric_obs(
        self,
        *names: str,
        cache: bool = False,
        convert: Converter | None = None,
    ) -> lsl.Calc | lsl.TransientCalc:
        """Column-stack numeric observed variables in a calculation node."""
        vars_ = [self.get_numeric_obs(name, convert=convert) for name in names]
        key = (names, cache)
        if key not in self._matrix_cache:
            node_type = lsl.Calc if cache else lsl.TransientCalc
            full_name = ",".join(self.prefix + name for name in names)
            self._matrix_cache[key] = node_type(
                lambda *args: jnp.vstack(args).T,
                *vars_,
                _name=f"[{full_name}]",
            )
        return self._matrix_cache[key]

    def get_calc(
        self,
        name: str,
        transform: Callable,
        var_name: str | None = None,
        cache_key: str | None = None,
        *,
        convert: Converter | None = None,
        cache: bool = True,
    ) -> lsl.Var:
        """Get a registry-cached calculation of an observed variable."""
        base_var = self.get_obs(name, convert=convert)
        func_hash = (
            cache_key if cache_key is not None else self._hash_function(transform)
        )
        if func_hash is None:
            return self._make_derived_var(base_var, transform, var_name, cache=cache)

        full_cache_key = (name, func_hash, var_name, cache)
        if full_cache_key not in self._derived_cache:
            self._derived_cache[full_cache_key] = self._make_derived_var(
                base_var, transform, var_name, cache=cache
            )
        return self._derived_cache[full_cache_key]

    def get_calc_centered(
        self,
        name: str,
        var_name: str | None = None,
        *,
        convert: Converter | None = None,
        cache: bool = True,
    ) -> lsl.Var:
        """Get a centered calculation of a numeric observed variable."""
        base_var = self.get_numeric_obs(name, convert=convert)
        mean = float(np.mean(base_var.value))

        def centered(value):
            return value - mean

        return self._make_derived_var(
            base_var, centered, var_name or f"{name}_centered", cache=cache
        )

    def get_calc_standardized(
        self,
        name: str,
        var_name: str | None = None,
        *,
        convert: Converter | None = None,
        cache: bool = True,
    ) -> lsl.Var:
        """Get a standardized calculation of a numeric observed variable."""
        base_var = self.get_numeric_obs(name, convert=convert)
        mean = float(np.mean(base_var.value))
        std = float(np.std(base_var.value))
        if std == 0:
            raise ValueError(
                "Failed to apply transformation 'standardization' to variable "
                f"'{name}': standard deviation is zero (constant variable)"
            )

        def standardized(value):
            return (value - mean) / std

        return self._make_derived_var(
            base_var, standardized, var_name or f"{name}_std", cache=cache
        )

    def get_calc_dummymatrix(
        self,
        name: str,
        var_name_prefix: str | None = None,
        *,
        cache: bool = True,
    ) -> lsl.Var:
        """Get a dummy-coded array with the category axis appended.

        An input with shape ``S`` returns shape ``S + (n_categories - 1,)``. The
        first category is the reference category.
        """
        base_var, mapping = self.get_categorical_obs(name)
        base_var.name = f"{name}_codes"
        n_categories = len(mapping.labels_to_integers_map)
        if n_categories < 2:
            raise ValueError(
                "Failed to apply transformation 'dummy encoding' to variable "
                f"'{name}': only {n_categories} unique value(s) found"
            )

        def dummy(codes):
            matrix = jnp.zeros((*codes.shape, n_categories - 1), dtype=jnp.float32)
            for index in range(1, n_categories):
                matrix = matrix.at[..., index - 1].set(codes == index)
            unknown = (codes >= n_categories) | (codes < 0)
            return jnp.where(unknown[..., None], jnp.nan, matrix)

        prefix = var_name_prefix or f"{name}_"
        return lsl.Var.new_calc(
            dummy,
            base_var,
            name=f"{prefix}matrix",
            cache=cache,
        )

    def observed_position(
        self, model: lsl.Model, data: Mapping[str, Any] | pd.DataFrame
    ) -> Position:
        """Encode the model's observed variables from a mapping.

        Registry-backed variables are resolved to their original source keys,
        including variables whose names have a registry prefix. Other observed
        variables are resolved by matching their model name to a key. Extra keys are
        ignored, and categorical values use the mapping established by the registry's
        setup data.

        Parameters
        ----------
        model
            Model whose observed variables define the position entries.
        data
            Source values to encode. Every observed variable required by ``model``
            must be resolvable.

        Notes
        -----
        The result contains raw observed values, not evaluated basis matrices or other
        derived model quantities.

        Examples
        --------
        >>> import liesel.model as lsl
        >>> from liesel_gam import DictRegistry
        >>> registry = DictRegistry(
        ...     {"x": [1.0, 2.0], "group": ["a", "b"]},
        ...     prefix_names_by="loc.",
        ... )
        >>> x = registry.get_obs("x")
        >>> group, _ = registry.get_categorical_obs("group")
        >>> model = lsl.Model([x, group])
        >>> position = registry.observed_position(
        ...     model,
        ...     {"x": [3.0, 4.0], "group": ["b", "a"], "unused": [5.0, 6.0]},
        ... )
        >>> {name: value.tolist() for name, value in sorted(position.items())}
        {'loc.group': [1, 0], 'loc.x': [3.0, 4.0]}

        See Also
        --------
        liesel_gam.category_coverage_indices
            Find categorical coverage rows to reserve in training.
        liesel_gam.basis_setup_sample
            Draw a representative setup sample from selected training rows.
        :doc:`Large-data model setup </notebooks_large_data>`
            Compose the helpers in an executable workflow.
        """
        source_by_var = {var.name: name for name, var in self._var_cache.items()}
        position = {}
        unresolved = []
        for var_name in model.observed:
            if var_name in source_by_var:
                source_name = source_by_var[var_name]
            elif var_name in data:
                source_name = var_name
            else:
                unresolved.append(var_name)
                continue
            position[var_name] = data[source_name]

        if unresolved:
            raise KeyError(
                "Could not resolve observed model variables from data: "
                f"{sorted(unresolved)}"
            )
        return model.convert_position(position)


class PandasRegistry(DictRegistry):
    """A :class:`DictRegistry` with DataFrame missing-data handling and metadata.

    Missing-data handling is applied before custom value conversion. Unlike
    :class:`DictRegistry`, this class retains ``columns`` and ``shape`` attributes.

    Parameters
    ----------
    data
        Source DataFrame.
    na_action
        Whether to reject, drop, or retain rows containing missing values.
    prefix_names_by
        Prefix for generated Liesel variable names.
    convert
        Default Liesel value converter; see :class:`DictRegistry` for converter
        semantics and the compiled-update warning.
    """

    data: pd.DataFrame

    def __init__(
        self,
        data: pd.DataFrame,
        na_action: Literal["error", "drop", "ignore"] = "error",
        prefix_names_by: str = "",
        convert: Converter = "default",
    ) -> None:
        """Initialize the registry from a pandas DataFrame."""
        if na_action not in ["error", "drop", "ignore"]:
            raise ValueError("na_action must be 'error', 'drop', or 'ignore'")

        self.original_data = data.copy()
        self.na_action = na_action
        clean_data = self._validate_data(data)
        super().__init__(
            {name: clean_data[name] for name in clean_data.columns},
            prefix_names_by=prefix_names_by,
            convert=convert,
        )
        self.data = clean_data

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.isna().any().any():
            if self.na_action == "error":
                na_cols = data.columns[data.isna().any()].tolist()
                raise ValueError(
                    f"Data contains NaN values in columns: {na_cols}. "
                    "Use na_action='drop' to automatically remove rows with NaN values."
                )
            if self.na_action == "drop":
                clean_data = data.dropna()
                if len(clean_data) == 0:
                    raise ValueError("No rows remaining after dropping NaN values")
                return clean_data
            if self.na_action == "ignore":
                pass
            else:
                assert_never(self.na_action)
        return data.copy()

    @property
    def columns(self) -> list[str]:
        """Column names after missing-data handling."""
        return list(self.data.columns)

    @property
    def shape(self) -> tuple[int, int]:
        """DataFrame shape after missing-data handling."""
        return self.data.shape

    def observed_position(
        self, model: lsl.Model, data: Mapping[str, Any] | pd.DataFrame
    ) -> Position:
        """Encode observed values, rejecting missing or non-finite DataFrame entries.

        Parameters
        ----------
        model
            Model whose observed variables define the position entries.
        data
            Mapping or DataFrame containing the source values to encode.
        """
        if not isinstance(data, pd.DataFrame):
            return super().observed_position(model, data)

        source_by_var = {var.name: name for name, var in self._var_cache.items()}
        for var_name in model.observed:
            source_name = source_by_var.get(var_name, var_name)
            if source_name not in data:
                continue
            values = data[source_name]
            is_nonfinite = (
                pd.api.types.is_numeric_dtype(values)
                and not np.isfinite(values.to_numpy()).all()
            )
            if values.isna().any() or is_nonfinite:
                raise ValueError(
                    f"Observed column {source_name!r} contains missing or non-finite "
                    "values."
                )
        return super().observed_position(model, data)
