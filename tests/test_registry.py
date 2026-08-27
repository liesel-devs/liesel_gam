"""Tests for VariableRegistry."""

import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest

import liesel_gam as gam
from liesel_gam.registry import DictRegistry, PandasRegistry


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
    return PandasRegistry(sample_data)


def test_dict_registry_gets_and_caches_observed_values():
    registry = DictRegistry({"x": [1.0, 2.0]}, prefix_names_by="data.")

    first = registry.get_obs("x")

    assert first is registry.get_obs("x")
    assert first.name == "data.x"
    assert first.value.tolist() == [1.0, 2.0]


def test_pandas_registry_is_a_dict_registry(sample_data):
    registry = PandasRegistry(sample_data)

    assert isinstance(registry, DictRegistry)


def test_pandas_registry_supports_default_value_conversion():
    registry = PandasRegistry(
        pd.DataFrame({"x": ["1", "2"]}),
        convert=lambda value: jnp.asarray(value, dtype=float),
    )

    assert registry.get_numeric_obs("x").value.tolist() == [1.0, 2.0]


def test_dict_registry_copies_mapping_but_reflects_data_mutation():
    source = {"x": [1.0]}
    registry = DictRegistry(source)
    source["y"] = [2.0]
    registry.data["z"] = [3.0]

    assert list(registry.keys()) == ["x", "z"]
    assert not hasattr(registry, "shape")
    assert registry.get_obs("z").value.tolist() == [3.0]


def test_dict_registry_cached_variable_survives_data_mutation():
    registry = DictRegistry({"x": [1.0]})
    first = registry.get_obs("x")
    registry.data["x"] = [2.0]

    assert registry.get_obs("x") is first
    assert first.value.tolist() == [1.0]


def test_dict_registry_rejects_unsupported_sources():
    with pytest.raises(TypeError, match="keys must be strings"):
        DictRegistry({1: [1.0]})  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="must not be Liesel"):
        DictRegistry({"x": lsl.Var.new_obs([1.0])})


def test_dict_registry_inherits_and_overrides_value_conversion():
    def as_float(value):
        return jnp.asarray(value, dtype=float)

    def as_int(value):
        return jnp.asarray(value, dtype=int)

    registry = DictRegistry({"x": [1], "y": [1], "z": [1]}, convert=as_float)

    assert jnp.issubdtype(registry.get_obs("x").value.dtype, jnp.floating)
    assert jnp.issubdtype(
        registry.get_obs("y", convert=as_int).value.dtype, jnp.integer
    )
    assert registry.get_obs("z", convert="default").value.tolist() == [1]


def test_dict_registry_rejects_a_different_converter_after_materialization():
    def converter(value):
        return jnp.asarray(value)

    registry = DictRegistry({"x": [1.0]}, convert=converter)
    registry.get_obs("x")

    assert registry.get_obs("x", convert=converter) is registry.get_obs("x")
    with pytest.raises(ValueError, match="different converter"):
        registry.get_obs("x", convert="default")


def test_dict_registry_compares_callable_converters_by_identity():
    class Converter:
        def __call__(self, value):
            return jnp.asarray(value)

        def __eq__(self, other):
            return True

    first = Converter()
    registry = DictRegistry({"x": [1.0]}, convert=first)
    registry.get_obs("x")

    assert registry.get_obs("x", convert=first) is registry.get_obs("x")
    with pytest.raises(ValueError, match="different converter"):
        registry.get_obs("x", convert=Converter())


def test_dict_registry_type_checks_converted_values():
    registry = DictRegistry(
        {"number": ["1", "2"], "flag": [0, 1]},
        convert=lambda value: jnp.asarray(value, dtype=float),
    )

    assert registry.get_numeric_obs("number").value.tolist() == [1.0, 2.0]
    assert registry.get_boolean_obs(
        "flag", convert=lambda value: jnp.asarray(value, dtype=bool)
    ).value.tolist() == [False, True]


def test_dict_registry_categorical_detection_supports_arbitrary_shapes():
    registry = DictRegistry(
        {
            "strings": np.array([["b", "a"], ["a", "b"]], dtype=object),
            "numeric_categories": pd.Categorical([2, 1], categories=[1, 2]),
            "numbers": np.array([1, 2]),
        }
    )

    strings, _ = registry.get_categorical_obs("strings")

    assert isinstance(strings, gam.CatVar)
    assert strings.value.tolist() == [[1, 0], [0, 1]]
    assert registry.is_categorical("numbers") is False
    with pytest.raises(TypeError, match="must not be integers"):
        registry.get_categorical_obs("numeric_categories")
    with pytest.raises(TypeError, match="expected categorical"):
        registry.get_categorical_obs("numbers")


def test_dict_registry_calculation_cache_mode_is_part_of_identity():
    registry = DictRegistry({"x": [1.0, 2.0]})

    persistent = registry.get_calc("x", jnp.square, cache=True)
    transient = registry.get_calc("x", jnp.square, cache=False)

    assert persistent is registry.get_calc("x", jnp.square, cache=True)
    assert transient is registry.get_calc("x", jnp.square, cache=False)
    assert persistent is not transient
    assert isinstance(persistent.value_node, lsl.Calc)
    assert isinstance(transient.value_node, lsl.TransientCalc)


def test_dict_registry_matrix_cache_mode_is_part_of_identity():
    registry = DictRegistry({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    persistent = registry.get_many_numeric_obs("x", "y", cache=True)
    transient = registry.get_many_numeric_obs("x", "y", cache=False)

    assert persistent is not transient
    assert persistent.value.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert isinstance(persistent, lsl.Calc)
    assert isinstance(transient, lsl.TransientCalc)


def test_dict_registry_centered_and_standardized_calculations_accept_options():
    def converter(value):
        return jnp.asarray(value, dtype=float)

    registry = DictRegistry({"x": ["1", "2", "3"]})

    centered = registry.get_calc_centered("x", convert=converter, cache=False)
    standardized = registry.get_calc_standardized("x", convert=converter, cache=False)

    assert centered.value.tolist() == [-1.0, 0.0, 1.0]
    assert jnp.std(standardized.value) == pytest.approx(1.0)
    assert isinstance(centered.value_node, lsl.TransientCalc)
    assert isinstance(standardized.value_node, lsl.TransientCalc)


def test_dict_registry_dummy_calculation_uses_catvar_and_cache_option():
    registry = DictRegistry({"group": [["a", "b"], ["b", "a"]]})

    dummy = registry.get_calc_dummymatrix("group", cache=False)

    assert dummy.value.tolist() == [[[0.0], [1.0]], [[1.0], [0.0]]]
    assert isinstance(dummy.value_node, lsl.TransientCalc)


def test_dict_registry_dummy_calculation_appends_axis_to_scalar():
    registry = DictRegistry({"group": ["a", "b"]})
    group, _ = registry.get_categorical_obs("group")
    group.value = "b"

    dummy = registry.get_calc_dummymatrix("group")

    assert dummy.value.shape == (1,)
    assert dummy.value.tolist() == [1.0]


def test_dict_registry_dummy_calculation_keeps_invalid_code_nan_behavior():
    registry = DictRegistry({"group": ["a", "b"]})
    dummy = registry.get_calc_dummymatrix("group")
    assert isinstance(dummy.value_node, lsl.Calc)

    result = dummy.value_node.function(jnp.array([[0, 2]]))

    assert result.shape == (1, 2, 1)
    assert result[0, 0, 0] == 0.0
    assert jnp.isnan(result[0, 1, 0])


def test_dict_registry_observed_position_uses_model_converters():
    def scale(value):
        array = jnp.asarray(value)
        return array if isinstance(value, jnp.ndarray) else array * 10

    registry = DictRegistry(
        {"x": [1.0, 2.0], "group": ["a", "b"]},
        prefix_names_by="data.",
        convert=scale,
    )
    x = registry.get_obs("x")
    group, _ = registry.get_categorical_obs("group")
    model = lsl.Model([x, group])

    position = registry.observed_position(model, {"x": [3.0, 4.0], "group": ["b", "a"]})

    assert position["data.x"].tolist() == [30.0, 40.0]
    assert position["data.group"].tolist() == [1, 0]


def test_dict_registry_observed_position_preserves_categorical_shape():
    registry = DictRegistry({"group": [["a", "b"], ["b", "a"]]})
    group, _ = registry.get_categorical_obs("group")
    model = lsl.Model(group)

    position = registry.observed_position(model, {"group": [["b", "a"], ["a", "b"]]})

    assert position["group"].tolist() == [[1, 0], [0, 1]]


def test_dict_registry_converter_can_support_compiled_repeated_conversion():
    def convert(value):
        if isinstance(value, list):
            return jnp.asarray(value, dtype=float) * 10
        return jnp.asarray(value, dtype=float)

    variable = DictRegistry({"x": [1.0, 2.0]}, convert=convert).get_obs("x")
    model = lsl.Model(variable)

    def update(value):
        state = model.update_state({"x": value})
        return model.extract_position(["x"], model_state=state)["x"]

    assert variable.value.tolist() == [10.0, 20.0]
    assert jax.jit(update)(jnp.array([30.0, 40.0])).tolist() == [30.0, 40.0]


def test_basic_get_var(sample_data):
    registry = PandasRegistry(sample_data)

    # get variable
    var1 = registry.get_obs("x1")
    assert var1.name == "x1"
    assert jnp.allclose(var1.value, sample_data["x1"].to_numpy())

    # test caching
    var2 = registry.get_obs("x1")
    assert var1 is var2


def test_observed_position_encodes_only_model_observations():
    setup = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "group": pd.Categorical(["a", "b"]),
            "y": [3.0, 4.0],
        }
    )
    registry = PandasRegistry(setup, prefix_names_by="loc.")
    x = registry.get_obs("x")
    group, _ = registry.get_categorical_obs("group")
    y = lsl.Var.new_obs(setup["y"].to_numpy(), name="y")
    model = lsl.Model([x, group, y])
    full = pd.DataFrame(
        {
            "x": [5.0, 6.0, 7.0],
            "group": ["b", "a", "b"],
            "y": [8.0, 9.0, 10.0],
            "unused": [11.0, 12.0, 13.0],
        }
    )

    position = registry.observed_position(model, full)

    assert set(position) == {"loc.x", "loc.group", "y"}
    assert position["loc.x"].tolist() == [5.0, 6.0, 7.0]
    assert position["loc.group"].tolist() == [1, 0, 1]
    assert position["y"].tolist() == [8.0, 9.0, 10.0]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_observed_position_rejects_missing_or_nonfinite_values(bad_value):
    setup = pd.DataFrame({"x": [1.0, 2.0]})
    registry = PandasRegistry(setup)
    model = lsl.Model([registry.get_obs("x")])

    with pytest.raises(ValueError, match="missing or non-finite"):
        registry.observed_position(model, pd.DataFrame({"x": [3.0, bad_value]}))


def test_variable_not_found(registry: PandasRegistry):
    with pytest.raises(KeyError):
        registry.get_obs("missing")


def test_centered_var(registry: PandasRegistry):
    centered = registry.get_calc_centered("x1")
    assert centered.name == "x1_centered"

    # check that mean is approximately zero
    assert jnp.mean(centered.value) == pytest.approx(0.0, abs=1e-8)


def test_std_var(registry: PandasRegistry):
    std_var = registry.get_calc_standardized("x1")
    assert std_var.name == "x1_std"

    # check standardization: mean ≈ 0, std ≈ 1
    assert jnp.mean(std_var.value) == pytest.approx(0.0, abs=1e-7)
    assert jnp.std(std_var.value) == pytest.approx(1.0)


def test_std_var_constant_error(registry: PandasRegistry):
    with pytest.raises(ValueError):
        registry.get_calc_standardized("x3")


def test_dummy_vars(registry: PandasRegistry):
    dummy_matrix = registry.get_calc_dummymatrix("cat")
    assert dummy_matrix.name == "cat_matrix"

    # should be (n_obs, n_categories-1) matrix
    assert dummy_matrix.value.shape == (50, 2)  # 3 categories - 1 reference

    # check that each row sums to 0 or 1
    row_sums = jnp.sum(dummy_matrix.value, axis=1)
    assert jnp.all((row_sums == 0) | (row_sums == 1))


def test_dummy_vars_type_error(registry: PandasRegistry):
    with pytest.raises(TypeError):
        registry.get_calc_dummymatrix("x1")


def test_dummy_vars_single_category_error(registry: PandasRegistry):
    with pytest.raises(ValueError):
        registry.get_calc_dummymatrix("single_cat")


def test_na_handling_error():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError):
        PandasRegistry(data, na_action="error")


def test_na_handling_drop():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    registry = PandasRegistry(data, na_action="drop")
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

    registry = PandasRegistry(data, na_action="ignore")
    assert registry.shape == (4, 2)  # no rows dropped

    # check that NaN row is still present
    var_x = registry.get_obs("x")
    assert jnp.isnan(var_x.value).any()


def test_properties(sample_data):
    registry = PandasRegistry(sample_data)

    assert registry.columns == list(sample_data.columns)
    assert registry.shape == sample_data.shape


def test_is_numeric(registry: PandasRegistry):
    assert registry.is_numeric("x1") is True
    assert registry.is_numeric("x2") is True
    assert registry.is_numeric("bool_var") is True
    assert registry.is_numeric("cat_str") is False
    assert registry.is_numeric("cat_num") is False


def test_is_categorical(registry: PandasRegistry):
    assert registry.is_categorical("cat_str") is True
    assert registry.is_categorical("cat_num") is True
    assert registry.is_categorical("cat") is True
    assert registry.is_categorical("x1") is False
    assert registry.is_categorical("bool_var") is False


def test_is_boolean(registry: PandasRegistry):
    assert registry.is_boolean("bool_var") is True
    assert registry.is_boolean("x1") is False
    assert registry.is_boolean("cat_str") is False


def test_type_check_nonexistent(registry: PandasRegistry):
    with pytest.raises(KeyError):
        registry.is_numeric("nonexistent")
    with pytest.raises(KeyError):
        registry.is_categorical("nonexistent")
    with pytest.raises(KeyError):
        registry.is_boolean("nonexistent")


def test_get_numeric_vars_success(registry: PandasRegistry):
    result = registry.get_numeric_obs("x1")
    assert result.name == "x1"


def test_get_numeric_var_failure(registry: PandasRegistry):
    with pytest.raises(TypeError):
        registry.get_numeric_obs("cat_str")
    with pytest.raises(TypeError):
        registry.get_numeric_obs("cat_num")


def test_get_categorical_var_success(registry: PandasRegistry):
    result, codes = registry.get_categorical_obs("cat_str")
    result_again, codes_again = registry.get_categorical_obs("cat_str")

    assert isinstance(result, gam.CatVar)
    assert result.name == "cat_str"
    assert result.mapping is codes
    assert result_again is result
    assert codes_again is codes
    assert codes.labels_to_integers_map == {"a": 0, "b": 1}
    assert codes.integers_to_labels_map == {0: "a", 1: "b"}

    computed_codes = codes.labels_to_integers(["a", "b"])
    assert np.all(computed_codes == np.array([0, 1]))

    computed_labels = codes.integers_to_labels([0, 1])
    assert np.all(computed_labels == np.array(["a", "b"]))

    with pytest.raises(TypeError, match="must not be integers"):
        registry.get_categorical_obs("cat_num")


def test_get_obs_and_mapping(registry: PandasRegistry):
    result = registry.get_obs_and_mapping("cat_str")
    assert result.var.name == "cat_str"
    assert result.mapping is not None
    assert result.mapping.labels_to_integers_map == {"a": 0, "b": 1}
    assert result.mapping.integers_to_labels_map == {0: "a", 1: "b"}

    result = registry.get_obs_and_mapping("x1")
    assert result.mapping is None
    assert result.var.name == "x1"

    result = registry.get_obs_and_mapping("bool_var")
    assert result.mapping is None
    assert result.var.name == "bool_var"


def test_get_categorical_var_failure(registry: PandasRegistry):
    with pytest.raises(TypeError):
        registry.get_categorical_obs("x1")


def test_get_boolean_var_success(registry: PandasRegistry):
    result = registry.get_boolean_obs("bool_var")
    assert result.name == "bool_var"


def test_get_boolean_var_failure(registry: PandasRegistry):
    with pytest.raises(TypeError):
        registry.get_boolean_obs("x1")


def test_get_calc_caching_simple_function(registry: PandasRegistry):
    def square(x):
        return x**2

    # first call should compute and cache
    result1 = registry.get_calc("x1", square)
    assert result1.name.startswith("x1_square")

    # second call should use cache (same object)
    result2 = registry.get_calc("x1", square)
    assert result1 is result2


def test_get_calc_caching_with_transformer_class(registry: PandasRegistry):
    class Transformer:
        def __init__(self, factor):
            self.factor = factor

        def __call__(self, x):
            return self.factor * x

        def more(self, x):
            return 2 * self.factor * x

    transformer = Transformer(2)
    # first call should compute and cache
    result1 = registry.get_calc("x1", transformer)

    # second call should use cache (same object)
    result2 = registry.get_calc("x1", transformer)
    assert result1 is result2

    # different transformer should create new variable
    transformer2 = Transformer(3)
    result3 = registry.get_calc("x1", transformer2)
    assert result3 is not result1

    # different method should not use same cache
    result4 = registry.get_calc("x1", transformer.more)
    assert result4 is not result1

    # but same method on same transformer should use cache
    result5 = registry.get_calc("x1", transformer.more)
    assert result5 is result4

    # different transformer with same method should create new variable
    result6 = registry.get_calc("x1", transformer2.more)
    assert result6 is not result4


def test_get_calc_explicit_cache_key(registry: PandasRegistry):
    def transform(x):
        return 2 * x

    # use explicit cache key
    result1 = registry.get_calc("x1", transform, cache_key="double")
    result2 = registry.get_calc("x1", transform, cache_key="double")

    # should be cached
    assert result1 is result2

    # different cache key should create new variable
    result3 = registry.get_calc("x1", transform, cache_key="different")
    assert result3 is not result1

    def transform2(x):
        return 2 * x

    # same cache key with different function should use cache
    result4 = registry.get_calc("x1", transform, cache_key="double")
    assert result4 is result1


def test_get_calc_closure_warning(registry: PandasRegistry):
    unsupported_data = {"key": "value"}  # dict is not supported

    def closure_func(x):
        return x + len(unsupported_data)

    # should issue warning and skip caching
    with pytest.warns(UserWarning, match="unsupported closure variable type"):
        result1 = registry.get_calc("x1", closure_func)
        result2 = registry.get_calc("x1", closure_func)

    # should compute each time (not cached)
    assert result1 is not result2


def test_get_calc_jax_closure_caching(registry: PandasRegistry):
    import jax.numpy as jnp

    # closures over jax arrays should cache correctly
    multiplier = jnp.array([2.0, 3.0])

    def jax_closure(x):
        return x * multiplier.sum()

    # should cache successfully
    result1 = registry.get_calc("x1", jax_closure)
    result2 = registry.get_calc("x1", jax_closure)

    assert result1 is result2

    # different multiplier should create different cache entry
    multiplier2 = jnp.array([4.0, 5.0])

    def jax_closure2(x):
        return x * multiplier2.sum()

    result3 = registry.get_calc("x1", jax_closure2)
    assert result3 is not result1


def test_get_calc_different_var_names(registry: PandasRegistry):
    def transform(x):
        return 2 * x

    result1 = registry.get_calc("x1", transform, var_name="triple1")
    result2 = registry.get_calc("x1", transform, var_name="triple2")

    # different var_names should create different variables
    assert result1 is not result2
    assert result1.name == "triple1"
    assert result2.name == "triple2"


def test_get_calc_cache_across_base_variables(registry: PandasRegistry):
    def transform(x):
        return x + 1

    result_x1 = registry.get_calc("x1", transform)
    result_x2 = registry.get_calc("x2", transform)

    # different base variables should create different cache entries
    assert result_x1 is not result_x2


def test_dummy_vars_unknown_category_values():
    """Categorical variables reject codes outside their mapping."""
    # create data with known categories A, B (codes 0, 1)
    data = pd.DataFrame({"cat": pd.Categorical(["A", "B", "A", "B"])})

    registry = PandasRegistry(data)

    # get the dummy matrix for the original data
    # also creates the base variable with codes
    original_dummy = registry.get_calc_dummymatrix("cat")

    # verify original behavior with codes 0, 1
    expected_original = jnp.array(
        [
            [0],  # A (code 0, reference category)
            [1],  # B (code 1)
            [0],  # A (code 0, reference category)
            [1],  # B (code 1)
        ]
    )
    assert jnp.array_equal(original_dummy.value, expected_original)

    # now simulate what happens when the base variable contains an unknown code
    base_var = original_dummy.value_node.inputs[0].var
    assert base_var is not None
    with pytest.raises(ValueError, match="Unknown integer codes"):
        base_var.value = jnp.array([0, 1, 0, 2])
