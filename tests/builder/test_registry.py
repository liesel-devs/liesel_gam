"""Tests for VariableRegistry."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from liesel_gam.builder import VariableRegistry
from liesel_gam.builder.errors import (
    TypeMismatchError,
    VariableNotFoundError,
    VariableTransformError,
)


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
        }
    )

    return data


def test_basic_get_var(sample_data):
    registry = VariableRegistry(sample_data)

    # get variable
    var1 = registry.get_var("x1")
    assert var1.name == "x1"
    assert jnp.allclose(var1.value, sample_data["x1"].to_numpy())

    # test caching
    var2 = registry.get_var("x1")
    assert var1 is var2


def test_variable_not_found(sample_data):
    registry = VariableRegistry(sample_data)

    with pytest.raises(VariableNotFoundError):
        registry.get_var("missing")


def test_centered_var(sample_data):
    """Test variable centering."""
    registry = VariableRegistry(sample_data)

    centered = registry.get_centered_var("x1")
    assert centered.name == "x1_centered"

    # check that mean is approximately zero
    assert jnp.mean(centered.value) == pytest.approx(0.0, abs=1e-8)


def test_std_var(sample_data):
    """Test variable standardization."""
    registry = VariableRegistry(sample_data)

    std_var = registry.get_std_var("x1")
    assert std_var.name == "x1_std"

    # check standardization: mean ≈ 0, std ≈ 1
    assert jnp.mean(std_var.value) == pytest.approx(0.0, abs=1e-7)
    assert jnp.std(std_var.value) == pytest.approx(1.0)


def test_std_var_constant_error(sample_data):
    registry = VariableRegistry(sample_data)

    with pytest.raises(VariableTransformError):
        registry.get_std_var("x3")


def test_dummy_vars(sample_data):
    registry = VariableRegistry(sample_data)

    dummy_matrix = registry.get_dummy_vars("cat")
    assert dummy_matrix.name == "cat_matrix"

    # should be (n_obs, n_categories-1) matrix
    assert dummy_matrix.value.shape == (50, 2)  # 3 categories - 1 reference

    # check that each row sums to 0 or 1 (one-hot encoding minus reference)
    row_sums = jnp.sum(dummy_matrix.value, axis=1)
    assert jnp.all((row_sums == 0) | (row_sums == 1))


def test_dummy_vars_type_error(sample_data):
    registry = VariableRegistry(sample_data)

    with pytest.raises(TypeMismatchError):
        registry.get_dummy_vars("x1")


def test_dummy_vars_single_category_error(sample_data):
    registry = VariableRegistry(sample_data)

    with pytest.raises(VariableTransformError):
        registry.get_dummy_vars("single_cat")


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
    var_x = registry.get_var("x")
    assert not jnp.isnan(var_x.value).any()


def test_properties(sample_data):
    registry = VariableRegistry(sample_data)

    assert registry.columns == list(sample_data.columns)
    assert registry.shape == sample_data.shape
