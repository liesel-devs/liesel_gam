"""Tests for GamBuilder class."""

import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_gam.builder.errors import JAXCompatibilityError
from liesel_gam.builder.gam_builder import GamBuilder
from liesel_gam.predictor import AdditivePredictor
from liesel_gam.var import Intercept, LinearTerm, SmoothTerm


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    rng = np.random.default_rng(42)
    n = 100

    data = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(-1, 1, n),
            "x3": rng.normal(2, 0.5, n),
            "y": rng.normal(0, 1, n),
            "category": pd.Categorical(
                rng.choice(["A", "B", "C"], size=n), categories=["A", "B", "C"]
            ),
        }
    )
    return data


@pytest.fixture
def builder(sample_data) -> GamBuilder:
    return GamBuilder(sample_data)


# Basic initialization tests
def test_init_basic(sample_data):
    builder = GamBuilder(sample_data)

    assert builder.registry is not None
    assert builder.parser is not None
    assert builder._name_counter == 0
    assert builder._current_predictor_prefix is None
    assert len(builder.registry.columns) == 5


def test_init_with_custom_handlers(sample_data):
    from liesel_gam.builder.components import MGCVComponent

    custom_handlers = {"custom_smooth": MGCVComponent}
    builder = GamBuilder(sample_data, handlers=custom_handlers)

    # Check that custom handlers were passed to parser
    assert "custom_smooth" in builder.parser.handlers
    assert builder.parser.handlers["custom_smooth"] == MGCVComponent


# Terms method tests
def test_terms_linear_only(sample_data):
    """Test terms method with linear terms only."""
    builder = GamBuilder(sample_data)

    terms = builder.terms("x1 + x2")

    assert len(terms) == 1
    assert isinstance(terms[0], LinearTerm)
    # Should not include intercept by default in terms()
    assert terms[0].basis.value.shape[1] == 2  # x1 + x2, no intercept

    terms = builder.terms("1 + x1 + x2")

    assert len(terms) == 1
    assert isinstance(terms[0], LinearTerm)
    # Should include intercept
    assert terms[0].basis.value.shape[1] == 3  # x1 + x2, no intercept


def test_terms_intercept_only(sample_data):
    builder = GamBuilder(sample_data)

    terms = builder.terms("1")

    assert len(terms) == 1
    assert isinstance(terms[0], Intercept)


def test_terms_mixed_with_smooth(sample_data):
    builder = GamBuilder(sample_data)

    terms = builder.terms("x1 + s(x2)")

    assert len(terms) == 2
    linear_term = next(t for t in terms if isinstance(t, LinearTerm))
    smooth_term = next(t for t in terms if isinstance(t, SmoothTerm))

    assert linear_term.basis.value.shape[1] == 1  # just x1, no intercept
    assert smooth_term is not None


def test_terms_empty_formula_error(sample_data):
    builder = GamBuilder(sample_data)

    with pytest.raises(ValueError, match="Formula cannot be empty"):
        builder.terms("")


# Predictor creation tests
def test_predictor_linear_only(sample_data):
    builder = GamBuilder(sample_data)

    predictor = builder.predictor("x1 + x2")

    assert isinstance(predictor, AdditivePredictor)
    assert len(predictor.terms) == 1  # one LinearTerm with intercept included

    # Check that the linear term includes intercept
    linear_term = list(predictor.terms.values())[0]
    assert isinstance(linear_term, LinearTerm)
    assert linear_term.basis.value.shape[1] == 3  # intercept + x1 + x2


def test_predictor_no_intercept(sample_data):
    builder = GamBuilder(sample_data)

    predictor = builder.predictor("0 + x1 + x2")

    assert isinstance(predictor, AdditivePredictor)
    assert len(predictor.terms) == 1

    linear_term = list(predictor.terms.values())[0]
    assert isinstance(linear_term, LinearTerm)
    assert linear_term.basis.value.shape[1] == 2  # x1 + x2, no intercept


def test_predictor_intercept_only(sample_data):
    builder = GamBuilder(sample_data)

    predictor = builder.predictor("1")

    assert isinstance(predictor, AdditivePredictor)
    assert len(predictor.terms) == 1

    intercept_term = list(predictor.terms.values())[0]
    assert isinstance(intercept_term, Intercept)


def test_predictor_with_smooth_liesel(sample_data):
    builder = GamBuilder(sample_data)

    predictor = builder.predictor("x1 + ls(x2, k=15)")

    assert isinstance(predictor, AdditivePredictor)
    assert len(predictor.terms) == 2  # LinearTerm + SmoothTerm

    terms = list(predictor.terms.values())
    linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
    smooth_terms = [t for t in terms if isinstance(t, SmoothTerm)]

    assert len(linear_terms) == 1
    assert len(smooth_terms) == 1


def test_predictor_with_smooth_mgcv(sample_data):
    builder = GamBuilder(sample_data)

    predictor = builder.predictor("x1 + s(x2, k=10)")

    assert isinstance(predictor, AdditivePredictor)
    assert len(predictor.terms) == 2  # LinearTerm + SmoothTerm

    terms = list(predictor.terms.values())
    linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
    smooth_terms = [t for t in terms if isinstance(t, SmoothTerm)]

    assert len(linear_terms) == 1
    assert len(smooth_terms) == 1


def test_predictor_custom_name(sample_data):
    builder = GamBuilder(sample_data)

    predictor = builder.predictor("x1 + x2", name="mu")

    assert predictor.name == "mu"


def test_predictor_empty_formula_error(sample_data):
    builder = GamBuilder(sample_data)

    with pytest.raises(ValueError):
        builder.predictor("")


def test_predictor_unknown_function(sample_data):
    builder = GamBuilder(sample_data)

    with pytest.raises(ValueError):
        builder.predictor("x1 + unknown(x2)")


# Response creation tests
def test_response_basic(sample_data):
    builder = GamBuilder(sample_data)
    predictor = builder.predictor("x1 + x2")

    response = builder.response("y", tfd.Normal, loc=predictor, scale=1.0)

    assert response.name == "y"
    assert np.allclose(response.value, sample_data["y"].to_numpy())
    assert response.dist_node is not None


def test_response_custom_name(sample_data):
    """Test creating response with custom name."""
    builder = GamBuilder(sample_data)
    predictor = builder.predictor("x1")

    response = builder.response(
        "y", tfd.Normal, loc=predictor, scale=1.0, name="response"
    )

    assert response.name == "response"


def test_response_missing_variable(sample_data):
    """Test error when response variable not in data."""
    builder = GamBuilder(sample_data)
    predictor = builder.predictor("x1")

    with pytest.raises(ValueError):
        builder.response("missing", tfd.Normal, loc=predictor, scale=1.0)


# Naming tests
def test_unique_naming(sample_data):
    """Test unique name generation."""
    builder = GamBuilder(sample_data)

    # Create multiple predictors without custom names
    pred1 = builder.predictor("x1")
    pred2 = builder.predictor("x2")

    assert pred1.name != pred2.name
    assert "predictor_1" in pred1.name
    assert "predictor_2" in pred2.name


def test_semantic_naming_with_prefix(sample_data):
    builder = GamBuilder(sample_data)

    predictor = builder.predictor("x1 + ls(x2)", name="mu")

    # Check that term names include the predictor prefix
    terms = list(predictor.terms.values())
    for term in terms:
        assert "mu" in term.name


# Integration tests
def test_end_to_end_workflow(sample_data):
    """Test complete workflow matching expected usage patterns."""
    builder = GamBuilder(sample_data)

    # Create predictor with mixed terms
    predictor = builder.predictor("x1 + ls(x2, k=15) + s(x3, k=10)")

    # Create response
    response = builder.response("y", tfd.Normal, loc=predictor, scale=1.0)

    # Verify everything works
    assert isinstance(predictor, AdditivePredictor)
    terms = list(predictor.terms.values())
    assert len(terms) == 3  # LinearTerm + 2 SmoothTerms

    linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
    smooth_terms = [t for t in terms if isinstance(t, SmoothTerm)]

    assert len(linear_terms) == 1
    assert len(smooth_terms) == 2

    assert response.name == "y"
    assert response.dist_node is not None


def test_nested_parentheses_in_formula(sample_data):
    builder = GamBuilder(sample_data)
    builder.predictor("s(x1, name='m(x)')", name="mu")


# Test use of categroical variable
def test_categorical_response(builder: GamBuilder):
    with pytest.raises(TypeError):
        builder.response("category", tfd.Normal, loc=0.0, scale=1.0)


def test_categorical_term(builder: GamBuilder):
    with pytest.raises(JAXCompatibilityError):
        builder.predictor("0 + category")


@pytest.mark.skip("TODO: should an error be raised")
def test_numerical_categorical_term():
    builder2 = GamBuilder(
        pd.DataFrame({"a": pd.Categorical([1, 2, 3]), "b": np.array([1.0, 2.0, 3.0])})
    )
    with pytest.raises(JAXCompatibilityError):
        builder2.terms("b")

    with pytest.raises(JAXCompatibilityError):
        builder2.terms("b")
