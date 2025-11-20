import jax.numpy as jnp
import numpy as np
import pandas as pd

import liesel_gam as gam

df = pd.DataFrame(
    {
        "test_string": ["a", "b", "c"],
        "test_np_array": np.array([1.0, 2.0, 3.0]),
        "test_np_string_array": np.array(["a", "b", "c"]),
        "test_jnp_array": jnp.array([1.0, 2.0, 3.0]),
        "test_integer": [1, 2, 3],
        "test_float": [1.0, 2.0, 3.0],
        "test_boolean": [True, False, True],
        "test_categorical": pd.Categorical(["a", "b", "c"]),
    }
)


class TestIsCategorical:
    def test_string(self) -> None:
        assert gam.series_is_categorical(df["test_string"])

    def test_np_array(self) -> None:
        assert not gam.series_is_categorical(df["test_np_array"])

    def test_np_string_array(self) -> None:
        assert gam.series_is_categorical(df["test_np_string_array"])

    def test_jnp_array(self) -> None:
        assert not gam.series_is_categorical(df["test_jnp_array"])

    def test_integer(self) -> None:
        assert not gam.series_is_categorical(df["test_integer"])

    def test_float(self) -> None:
        assert not gam.series_is_categorical(df["test_float"])

    def test_boolean(self) -> None:
        assert not gam.series_is_categorical(df["test_boolean"])

    def test_pandas_categorical(self) -> None:
        assert gam.series_is_categorical(df["test_categorical"])


class TestIsCategoricalEdgeCases:
    def test_pandas_categorical_numeric(self) -> None:
        s = pd.Series(pd.Categorical([1, 2, 1]))
        assert gam.series_is_categorical(s)

    def test_pandas_categorical_ordered(self) -> None:
        s = pd.Series(pd.Categorical(["low", "med", "high"], ordered=True))
        assert gam.series_is_categorical(s)

    def test_string_with_missing_values(self) -> None:
        s = pd.Series(["a", "b", None])
        assert gam.series_is_categorical(s)

    def test_categorical_with_missing_values(self) -> None:
        s = pd.Series(pd.Categorical(["a", None, "b"], categories=["a", "b"]))
        assert gam.series_is_categorical(s)

    def test_object_series_mixed_types(self) -> None:
        s = pd.Series(["a", 1, "b"])
        assert gam.series_is_categorical(s)

    def test_datetime_series_is_not_categorical(self) -> None:
        s = pd.Series(pd.date_range("2020-01-01", periods=3))
        assert not gam.series_is_categorical(s)

    def test_timedelta_series_is_not_categorical(self) -> None:
        s = pd.Series(pd.to_timedelta([1, 2, 3], unit="D"))
        assert not gam.series_is_categorical(s)

    def test_nullable_integer_series_is_not_categorical(self) -> None:
        s = pd.Series([1, 2, None], dtype="Int64")
        assert not gam.series_is_categorical(s)

    def test_boolean_nullable_series_is_not_categorical(self) -> None:
        s = pd.Series([True, False, None], dtype="boolean")
        assert not gam.series_is_categorical(s)

    def test_empty_string_series(self) -> None:
        s = pd.Series([], dtype=object)
        assert gam.series_is_categorical(s)

    def test_single_level_categorical(self) -> None:
        s = pd.Series(pd.Categorical(["a", "a", "a"]))
        assert gam.series_is_categorical(s)
