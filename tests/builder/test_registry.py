"""Tests for VariableRegistry."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from liesel_gam.builder import VariableRegistry


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    rng = np.random.default_rng(42)
    n = 50

    data = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(-1, 1, n),
            "x3": np.ones(n) * 2.5,  # constant variable
            "cat": pd.Categorical(["A", "B", "C"] * (n // 3) + ["A"] * (n % 3)),
            "single_cat": pd.Categorical(["X"] * n),
            "cat_str": pd.Categorical(["a", "b"] * (n // 2)),
            "cat_num": pd.Categorical([1, 2] * (n // 2)),
            "bool_var": [True, False] * (n // 2),
        }
    )

    return data


@pytest.fixture
def registry(sample_data):
    return VariableRegistry(sample_data)


def test_basic_get_var(sample_data):
    registry = VariableRegistry(sample_data)

    # get variable
    var1 = registry.get_obs("x1")
    assert var1.name == "x1"
    assert jnp.allclose(var1.value, sample_data["x1"].to_numpy())

    # test caching
    var2 = registry.get_obs("x1")
    assert var1 is var2


def test_variable_not_found(registry: VariableRegistry):
    with pytest.raises(KeyError):
        registry.get_obs("missing")


def test_centered_var(registry: VariableRegistry):
    centered = registry.get_centered_obs("x1")
    assert centered.name == "x1_centered"

    # check that mean is approximately zero
    assert jnp.mean(centered.value) == pytest.approx(0.0, abs=1e-8)


def test_std_var(registry: VariableRegistry):
    std_var = registry.get_standardized_obs("x1")
    assert std_var.name == "x1_std"

    # check standardization: mean ≈ 0, std ≈ 1
    assert jnp.mean(std_var.value) == pytest.approx(0.0, abs=1e-7)
    assert jnp.std(std_var.value) == pytest.approx(1.0)


def test_std_var_constant_error(registry: VariableRegistry):
    with pytest.raises(ValueError):
        registry.get_standardized_obs("x3")


def test_dummy_vars(registry: VariableRegistry):
    dummy_matrix = registry.get_dummy_obs("cat")
    assert dummy_matrix.name == "cat_matrix"

    # should be (n_obs, n_categories-1) matrix
    assert dummy_matrix.value.shape == (50, 2)  # 3 categories - 1 reference

    # check that each row sums to 0 or 1
    row_sums = jnp.sum(dummy_matrix.value, axis=1)
    assert jnp.all((row_sums == 0) | (row_sums == 1))


def test_dummy_vars_type_error(registry: VariableRegistry):
    with pytest.raises(TypeError):
        registry.get_dummy_obs("x1")


def test_dummy_vars_single_category_error(registry: VariableRegistry):
    with pytest.raises(ValueError):
        registry.get_dummy_obs("single_cat")


def test_na_handling_error():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError):
        VariableRegistry(data, na_action="error")


def test_na_handling_drop():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    registry = VariableRegistry(data, na_action="drop")
    assert registry.shape == (3, 2)  # one row dropped

    # check that NaN row was removed
    var_x = registry.get_obs("x")
    assert not jnp.isnan(var_x.value).any()


def test_na_handling_ignore():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    registry = VariableRegistry(data, na_action="ignore")
    assert registry.shape == (4, 2)  # no rows dropped

    # check that NaN row is still present
    var_x = registry.get_obs("x")
    assert jnp.isnan(var_x.value).any()


def test_properties(sample_data):
    registry = VariableRegistry(sample_data)

    assert registry.columns == list(sample_data.columns)
    assert registry.shape == sample_data.shape


def test_is_numeric(registry: VariableRegistry):
    assert registry.is_numeric("x1") is True
    assert registry.is_numeric("x2") is True
    assert registry.is_numeric("bool_var") is True
    assert registry.is_numeric("cat_str") is False
    assert registry.is_numeric("cat_num") is False


def test_is_categorical(registry: VariableRegistry):
    assert registry.is_categorical("cat_str") is True
    assert registry.is_categorical("cat_num") is True
    assert registry.is_categorical("cat") is True
    assert registry.is_categorical("x1") is False
    assert registry.is_categorical("bool_var") is False


def test_is_boolean(registry: VariableRegistry):
    assert registry.is_boolean("bool_var") is True
    assert registry.is_boolean("x1") is False
    assert registry.is_boolean("cat_str") is False


def test_type_check_nonexistent(registry: VariableRegistry):
    with pytest.raises(KeyError):
        registry.is_numeric("nonexistent")
    with pytest.raises(KeyError):
        registry.is_categorical("nonexistent")
    with pytest.raises(KeyError):
        registry.is_boolean("nonexistent")


def test_get_numeric_vars_success(registry: VariableRegistry):
    result = registry.get_numeric_obs("x1")
    assert result.name == "x1"


def test_get_numeric_var_failure(registry: VariableRegistry):
    with pytest.raises(TypeError):
        registry.get_numeric_obs("cat_str")
    with pytest.raises(TypeError):
        registry.get_numeric_obs("cat_num")


def test_get_categorical_var_success(registry: VariableRegistry):
    result, codes = registry.get_categorical_obs("cat_str")
    assert result.name == "cat_str"
    assert codes == {
        0: "a",
        1: "b",
    }

    result2, codes2 = registry.get_categorical_obs("cat_num")
    assert result2.name == "cat_num"
    assert codes2 == {0: 1, 1: 2}


def test_get_categorical_var_failure(registry: VariableRegistry):
    with pytest.raises(TypeError):
        registry.get_categorical_obs("x1")


def test_get_boolean_var_success(registry: VariableRegistry):
    result = registry.get_boolean_obs("bool_var")
    assert result.name == "bool_var"


def test_get_boolean_var_failure(registry: VariableRegistry):
    with pytest.raises(TypeError):
        registry.get_boolean_obs("x1")


def test_get_derived_obs_caching_simple_function(registry: VariableRegistry):
    def square(x):
        return x**2

    # first call should compute and cache
    result1 = registry.get_derived_obs("x1", square)
    assert result1.name.startswith("x1_square")

    # second call should use cache (same object)
    result2 = registry.get_derived_obs("x1", square)
    assert result1 is result2


def test_get_derived_obs_caching_with_transformer_class(registry: VariableRegistry):
    class Transformer:
        def __init__(self, factor):
            self.factor = factor

        def __call__(self, x):
            return self.factor * x

        def more(self, x):
            return 2 * self.factor * x

    transformer = Transformer(2)
    # first call should compute and cache
    result1 = registry.get_derived_obs("x1", transformer)

    # second call should use cache (same object)
    result2 = registry.get_derived_obs("x1", transformer)
    assert result1 is result2

    # different transformer should create new variable
    transformer2 = Transformer(3)
    result3 = registry.get_derived_obs("x1", transformer2)
    assert result3 is not result1

    # different method should not use same cache
    result4 = registry.get_derived_obs("x1", transformer.more)
    assert result4 is not result1

    # but same method on same transformer should use cache
    result5 = registry.get_derived_obs("x1", transformer.more)
    assert result5 is result4

    # different transformer with same method should create new variable
    result6 = registry.get_derived_obs("x1", transformer2.more)
    assert result6 is not result4


def test_get_derived_obs_explicit_cache_key(registry: VariableRegistry):
    def transform(x):
        return 2 * x

    # use explicit cache key
    result1 = registry.get_derived_obs("x1", transform, cache_key="double")
    result2 = registry.get_derived_obs("x1", transform, cache_key="double")

    # should be cached
    assert result1 is result2

    # different cache key should create new variable
    result3 = registry.get_derived_obs("x1", transform, cache_key="different")
    assert result3 is not result1


def test_get_derived_obs_closure_warning(registry: VariableRegistry):
    unsupported_data = {"key": "value"}  # dict is not supported

    def closure_func(x):
        return x + len(unsupported_data)

    # should issue warning and skip caching
    with pytest.warns(UserWarning, match="unsupported closure variable type"):
        result1 = registry.get_derived_obs("x1", closure_func)
        result2 = registry.get_derived_obs("x1", closure_func)

    # should compute each time (not cached)
    assert result1 is not result2


def test_get_derived_obs_jax_closure_caching(registry: VariableRegistry):
    import jax.numpy as jnp

    # closures over jax arrays should cache correctly
    multiplier = jnp.array([2.0, 3.0])

    def jax_closure(x):
        return x * multiplier.sum()

    # should cache successfully
    result1 = registry.get_derived_obs("x1", jax_closure)
    result2 = registry.get_derived_obs("x1", jax_closure)

    assert result1 is result2

    # different multiplier should create different cache entry
    multiplier2 = jnp.array([4.0, 5.0])

    def jax_closure2(x):
        return x * multiplier2.sum()

    result3 = registry.get_derived_obs("x1", jax_closure2)
    assert result3 is not result1


def test_get_derived_obs_different_var_names(registry: VariableRegistry):
    def transform(x):
        return 2 * x

    result1 = registry.get_derived_obs("x1", transform, var_name="triple1")
    result2 = registry.get_derived_obs("x1", transform, var_name="triple2")

    # different var_names should create different variables
    assert result1 is not result2
    assert result1.name == "triple1"
    assert result2.name == "triple2"


def test_get_derived_obs_cache_across_base_variables(registry: VariableRegistry):
    def transform(x):
        return x + 1

    result_x1 = registry.get_derived_obs("x1", transform)
    result_x2 = registry.get_derived_obs("x2", transform)

    # different base variables should create different cache entries
    assert result_x1 is not result_x2
