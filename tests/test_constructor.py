import jax
import jax.numpy as jnp
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n = 100
    rng = jax.random.key(42)
    rng1, rng2 = jax.random.split(rng, 2)

    x1 = jnp.linspace(0, 1, n) ** 2
    x2 = jnp.linspace(-1, 1, n)
    x3 = jnp.sin(2 * jnp.pi * x1) + 0.1 * jax.random.normal(rng1, shape=(n,))
    y = 1.0 + 2.0 * x1 + 0.5 * x2 + x3 + 0.2 * jax.random.normal(rng2, shape=(n,))

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def test_init(sample_data):
    """Test basic initialization."""
    constructor = gam.Constructor(sample_data)
    assert constructor.data is sample_data
    assert "s" in constructor.handlers
    assert "ls" in constructor.handlers


def test_init_with_custom_handlers(sample_data):
    """Test initialization with custom handlers."""

    def custom_handler(var_name, **kwargs):
        return gam.Intercept(f"custom_{var_name}")

    constructor = gam.Constructor(sample_data, {"foo": custom_handler})
    assert "foo" in constructor.handlers
    assert "s" in constructor.handlers
    assert "ls" in constructor.handlers


def test_get_var_caching(sample_data):
    """Test that variables are cached properly."""
    constructor = gam.Constructor(sample_data)

    var1 = constructor._get_obs("x1")
    var2 = constructor._get_obs("x1")

    assert var1 is var2
    assert var1.name == "x1"
    assert jnp.allclose(var1.value, sample_data["x1"].to_numpy())


def test_get_var_missing_column(sample_data):
    """Test error when variable not in data."""
    constructor = gam.Constructor(sample_data)

    with pytest.raises(ValueError, match="Variable 'missing' not found in data"):
        constructor._get_obs("missing")


def test_parse_formula_simple(sample_data):
    """Test parsing simple formula."""
    constructor = gam.Constructor(sample_data)

    components = constructor.parse_formula("x1 + x2", default_intercept=True)

    assert len(components) == 1
    linear_comp = components[0]
    assert isinstance(linear_comp, gam.constructor.LinearComponent)
    assert linear_comp.variables == ["x1", "x2"]
    assert linear_comp.include_intercept is True

    components = constructor.parse_formula("x1 + x2", default_intercept=False)
    linear_comp = components[0]
    assert linear_comp.include_intercept is False


def test_parse_formula_with_intercept_specification(sample_data):
    """Test parsing formula with intercept specification."""
    constructor = gam.Constructor(sample_data)

    components = constructor.parse_formula("0 + x1 + x2", default_intercept=True)
    assert len(components) == 1
    linear_comp = components[0]
    assert isinstance(linear_comp, gam.constructor.LinearComponent)
    assert linear_comp.variables == ["x1", "x2"]
    assert linear_comp.include_intercept is False

    components = constructor.parse_formula("1 + x1 + x2", default_intercept=False)
    assert len(components) == 1
    linear_comp = components[0]
    assert linear_comp.variables == ["x1", "x2"]
    assert linear_comp.include_intercept is True


def test_parse_formula_with_smooth(sample_data):
    """Test parsing formula with smooth terms."""
    constructor = gam.Constructor(sample_data)

    components = constructor.parse_formula(
        "x1 + ls(x2, k=10, bs='ps', a=1.0)", default_intercept=True
    )

    assert len(components) == 2
    # components are sorted: LinearComponent first, then FunctionComponent
    linear_comp = components[0]
    func_comp = components[1]

    assert isinstance(linear_comp, gam.constructor.LinearComponent)
    assert linear_comp.variables == ["x1"]
    assert linear_comp.include_intercept is True

    assert isinstance(func_comp, gam.constructor.FunctionComponent)
    assert func_comp.function == "ls"
    assert func_comp.variable == "x2"
    assert func_comp.kwargs == {"k": 10, "bs": "ps", "a": 1.0}


def test_terms_method_linear_only(sample_data):
    """Test _terms method with linear terms only."""
    constructor = gam.Constructor(sample_data)

    term = constructor.term("x1 + x2")
    assert isinstance(term, gam.LinearTerm)
    # should have 3 columns: intercept + x1 + x2
    assert term.basis.value.shape[1] == 3

    term = constructor.term("0 + x1 + x2")
    assert isinstance(term, gam.LinearTerm)
    # should have 2 columns: x1 + x2 (no intercept due to "0 +")
    assert term.basis.value.shape[1] == 2


def test_terms_method_with_smooth(sample_data):
    """Test _terms method with smooth terms."""
    constructor = gam.Constructor(sample_data)

    terms = constructor._terms("x1 + ls(x2, k=10)", default_intercept=True)

    assert len(terms) == 2
    # Components are sorted
    assert isinstance(terms[0], gam.LinearTerm)
    assert isinstance(terms[1], gam.SmoothTerm)
    # linear term should have intercept + x1
    assert terms[0].basis.value.shape[1] == 2


def test_terms_method_intercept_only(sample_data):
    """Test _terms method with intercept only."""
    constructor = gam.Constructor(sample_data)

    terms = constructor._terms("1", default_intercept=True)

    assert len(terms) == 1
    assert isinstance(terms[0], gam.Intercept)


def test_term_linear_multiple_vars(sample_data):
    """Test that term() allows multiple linear variables in one term."""
    constructor = gam.Constructor(sample_data)

    term = constructor.term("x1 + x2")  # multiple vars in one linear term

    assert isinstance(term, gam.LinearTerm)
    # should have 3 columns: intercept + x1 + x2
    assert term.basis.value.shape[1] == 3


def test_term_single_var(sample_data):
    """Test term() with single variable."""
    constructor = gam.Constructor(sample_data)

    term = constructor.term("x1")

    assert isinstance(term, gam.LinearTerm)
    # should have 2 columns: intercept + x1
    assert term.basis.value.shape[1] == 2


def test_term_smooth_only(sample_data):
    """Test term() with smooth term only."""
    constructor = gam.Constructor(sample_data)

    term = constructor.term("ls(x1, k=10)")

    assert isinstance(term, gam.SmoothTerm)
    assert term.name.startswith("ls(x1)_")


def test_term_multiple_terms_error(sample_data):
    """Test that term() rejects formulas with multiple additive terms."""
    constructor = gam.Constructor(sample_data)

    with pytest.raises(ValueError, match="term\\(\\) expects exactly one term"):
        constructor.term("x1 + ls(x2)")  # linear + smooth = 2 terms


def test_term_custom_name(sample_data):
    """Test term() with custom name."""
    constructor = gam.Constructor(sample_data)

    term = constructor.term("x1 + x2", name="custom_linear")

    assert isinstance(term, gam.LinearTerm)
    assert term.name == "custom_linear"


def test_predictor_linear_only(sample_data):
    """Test creating predictor with linear terms only."""
    constructor = gam.Constructor(sample_data)

    predictor = constructor.predictor("x1 + x2")

    assert isinstance(predictor, gam.AdditivePredictor)
    assert len(predictor.terms) == 1  # one LinearTerm with intercept included

    # check that the linear term includes intercept
    linear_term = list(predictor.terms.values())[0]
    assert isinstance(linear_term, gam.LinearTerm)
    assert linear_term.basis.value.shape[1] == 3


def test_predictor_no_intercept(sample_data):
    """Test creating predictor without intercept."""
    constructor = gam.Constructor(sample_data)

    predictor = constructor.predictor("0 + x1 + x2")

    assert isinstance(predictor, gam.AdditivePredictor)
    assert len(predictor.terms) == 1  # one LinearTerm without intercept

    linear_term = list(predictor.terms.values())[0]
    assert isinstance(linear_term, gam.LinearTerm)
    assert linear_term.basis.value.shape[1] == 2  # x1 + x2 only


def test_predictor_with_smooth_liesel(sample_data):
    """Test creating predictor with liesel spline smooth terms."""
    constructor = gam.Constructor(sample_data)

    # use ls() for liesel splines since smoothcon might not be available
    predictor = constructor.predictor("x1 + ls(x2, k=10)")

    assert isinstance(predictor, gam.AdditivePredictor)
    assert len(predictor.terms) == 2  # linearTerm + SmoothTerm

    # check we have the expected terms
    term0 = list(predictor.terms.values())[0]
    term1 = list(predictor.terms.values())[1]

    assert isinstance(term0, gam.LinearTerm)
    assert isinstance(term1, gam.SmoothTerm)


def test_predictor_custom_name(sample_data):
    """Test creating predictor with custom name."""
    constructor = gam.Constructor(sample_data)

    predictor = constructor.predictor("x1 + x2", name="custom_predictor")
    assert predictor.name == "custom_predictor"
    assert list(predictor.terms.values())[0].name == "custom_predictor_linear(x1+x2)"


def test_predictor_unknown_function(sample_data):
    """Test error with unknown function in formula."""
    constructor = gam.Constructor(sample_data)

    with pytest.raises(ValueError, match="Unknown function 'unknown'"):
        constructor.predictor("x1 + unknown(x2)")


def test_intercept_only_predictor(sample_data):
    """Test predictor with only intercept."""
    constructor = gam.Constructor(sample_data)

    predictor = constructor.predictor("1")  # just intercept

    assert isinstance(predictor, gam.AdditivePredictor)
    assert len(predictor.terms) == 1
    assert isinstance(list(predictor.terms.values())[0], gam.Intercept)


def test_intercept_empty_predictor(sample_data):
    """Test predictor with only intercept."""
    constructor = gam.Constructor(sample_data)

    with pytest.raises(ValueError, match="Predictor must have at least one term"):
        constructor.predictor("")


def test_response_basic(sample_data):
    """Test creating basic response variable."""
    constructor = gam.Constructor(sample_data)
    predictor = constructor.predictor("x1 + x2")

    response = constructor.response("y", tfd.Normal, loc=predictor, scale=1.0)

    assert response.name == "y"
    assert jnp.allclose(response.value, sample_data["y"].to_numpy())
    assert response.dist_node is not None


def test_response_missing_variable(sample_data):
    """Test error when response variable not in data."""
    constructor = gam.Constructor(sample_data)

    with pytest.raises(
        ValueError, match="Response variable 'missing' not found in data"
    ):
        constructor.response("missing", tfd.Normal, loc=0.0, scale=1.0)


def test_response_custom_name(sample_data):
    """Test creating response with custom name."""
    constructor = gam.Constructor(sample_data)
    predictor = constructor.predictor("x1 + x2")

    response = constructor.response(
        "y", tfd.Normal, name="custom_response", loc=predictor, scale=1.0
    )

    assert response.name == "custom_response"


def test_end_to_end_workflow(sample_data):
    """Test complete workflow matching examples in docstring."""
    constructor = gam.Constructor(sample_data)

    # create predictor
    predictor = constructor.predictor("x1 + ls(x2, k=15) + s(x3, k=5, bs='cr')")

    # create response
    y = constructor.response("y", tfd.Normal, loc=predictor, scale=1.0)

    # verify everything works
    assert isinstance(predictor, gam.AdditivePredictor)
    terms = list(predictor.terms.values())
    assert len(terms) == 3  # linearTerm + 2 SmoothTerms
    assert y.name == "y"
    assert y.dist_node is not None

    assert isinstance(terms[0], gam.LinearTerm)
    assert isinstance(terms[1], gam.SmoothTerm)
    assert isinstance(terms[2], gam.SmoothTerm)

    # linear term should have intercept
    assert terms[0].basis.value.shape[1] == 2


def test_manual_term_building_example(sample_data):
    """Test the manual term building example from user requirements."""
    constructor = gam.Constructor(sample_data)

    # this should work: combining manual AdditivePredictor with constructor terms
    scale = gam.AdditivePredictor("scale")
    scale += gam.Intercept("intercept")
    scale += constructor.term("x1 + x3")  # multiple linear vars in one term - OK
    scale += constructor.term("ls(x2)")  # single smooth term - OK

    assert isinstance(scale, gam.AdditivePredictor)
    assert len(scale.terms) == 3  # intercept + LinearTerm + SmoothTerm


def test_unique_naming(sample_data):
    """Test that Constructor generates unique names for auto-generated variables."""
    constructor = gam.Constructor(sample_data)

    # create multiple similar terms to test uniqueness
    term1 = constructor.term("ls(x1, k=10)")
    term2 = constructor.term("ls(x1, k=15)")  # same variable, different parameters
    predictor1 = constructor.predictor("x1 + x2")
    predictor2 = constructor.predictor("x1 + x2")  # same formula

    # all names should be different due to counter
    assert term1.name != term2.name
    assert predictor1.name != predictor2.name

    # names should have counter suffixes
    assert "_" in term1.name and term1.name.startswith("ls(x1)")
    assert "_" in term2.name and term2.name.startswith("ls(x1)")
    assert "_" in predictor1.name
    assert "_" in predictor2.name


def test_semantic_naming(sample_data):
    """Test semantic naming with predictor prefixes."""
    constructor = gam.Constructor(sample_data)

    # named predictor should use semantic naming
    predictor = constructor.predictor("x1 + ls(x2)", name="mu")

    # check that terms have predictor prefix
    term_names = list(predictor.terms.keys())
    assert any("mu_linear(x1)" in name for name in term_names)
    assert any("mu_ls(x2)" in name for name in term_names)


def test_get_term_keys(sample_data):
    """Test term key extraction for term_names mapping."""
    constructor = gam.Constructor(sample_data)

    keys = constructor.get_term_keys("x1 + x2 + ls(x3, k=10)")
    expected = ["linear(x1+x2)", "ls(x3)"]
    assert sorted(keys) == sorted(expected)

    keys = constructor.get_term_keys("s(x1) + ls(x2)")
    expected = ["s(x1)", "ls(x2)"]
    assert sorted(keys) == sorted(expected)


def test_custom_term_names(sample_data):
    """Test custom term naming via term_names parameter."""
    constructor = gam.Constructor(sample_data)

    predictor = constructor.predictor(
        "x1 + x2 + ls(x3)",
        name="mu",
        term_names={"linear(x1+x2)": "baseline", "ls(x3)": "trend"},
    )

    # check that custom names are used
    term_names = list(predictor.terms.keys())
    assert any("baseline" in name for name in term_names)
    assert any("trend" in name for name in term_names)
