"""Tests for formula parsing components."""

import numpy as np
import pandas as pd
import pytest

from liesel_gam.builder.components import (
    InterceptComponent,
    LieselSplineComponent,
    LinearComponent,
    MGCVComponent,
)
from liesel_gam.builder.formula import FormulaParser
from liesel_gam.builder.registry import VariableRegistry


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 50

    data = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(-1, 1, n),
            "x3": rng.normal(2, 0.5, n),
            "y": rng.normal(0, 1, n),
        }
    )
    return data


@pytest.fixture
def registry(sample_data):
    return VariableRegistry(sample_data)


# Parsing Tests
def test_parse_simple_linear_terms(registry):
    parser = FormulaParser(registry)

    # Single variable
    components = parser.parse("x1", default_intercept=False, merge=False)
    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, LinearComponent)
    assert comp.depends_on_vars == {"x1"}
    assert comp.includes_intercept is False

    # Multiple variables
    components = parser.parse("x1 + x2", default_intercept=False, merge=False)
    assert len(components) == 2
    comp = components[0]
    assert isinstance(comp, LinearComponent)
    comp = components[1]
    assert isinstance(comp, LinearComponent)


def test_parse_and_merge_linear_terms(registry):
    parser = FormulaParser(registry)
    components = parser.parse("x1 + x2", default_intercept=False, merge=False)
    merged_components = parser.merge_components(components)
    assert len(merged_components) == 1
    comp = merged_components[0]
    assert isinstance(comp, LinearComponent)
    assert comp.depends_on_vars == {"x1", "x2"}


def test_parse_function_calls(registry):
    parser = FormulaParser(registry)

    # LieselSpline function
    components = parser.parse("ls(x1)", default_intercept=False, merge=False)
    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, LieselSplineComponent)
    assert comp.func_name == "ls"
    assert comp.depends_on_vars == {"x1"}

    # MGCV function with parameters
    parser_with_handlers = FormulaParser(registry, handlers={"mgcv": MGCVComponent})
    components = parser_with_handlers.parse(
        "mgcv(x2, k=10)", default_intercept=False, merge=False
    )
    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, MGCVComponent)
    assert comp.func_name == "mgcv"
    assert comp.depends_on_vars == {"x2"}
    assert comp.mgcv_kwargs == {"k": 10}


def test_parse_explicit_intercept_specifications(registry):
    parser = FormulaParser(registry)

    # Explicit no intercept
    components = parser.parse("0", default_intercept=False, merge=False)
    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, InterceptComponent)
    assert comp.includes_intercept is False

    # Explicit intercept
    components = parser.parse("1", default_intercept=False, merge=False)
    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, InterceptComponent)
    assert comp.includes_intercept is True

    # No intercept with variable
    components = parser.parse("0 + x1", default_intercept=False, merge=False)
    assert len(components) == 2
    comp = components[0]
    assert isinstance(comp, InterceptComponent)
    assert comp.includes_intercept is False
    comp = components[1]
    assert isinstance(comp, LinearComponent)
    assert comp.depends_on_vars == {"x1"}
    assert comp.includes_intercept is False

    # Intercept with variable
    components = parser.parse("1 + x2", default_intercept=False, merge=False)
    assert len(components) == 2
    comp = components[0]
    assert isinstance(comp, InterceptComponent)
    assert comp.includes_intercept is True
    comp = components[1]
    assert isinstance(comp, LinearComponent)
    assert comp.depends_on_vars == {"x2"}
    assert comp.includes_intercept is False


def test_parse_mixed_formulas(registry):
    parser = FormulaParser(registry)
    components = parser.parse(
        "x1 + ls(x2) + s(x3)", default_intercept=False, merge=False
    )

    assert len(components) == 3

    # Should have one linear component and two function components
    linear_comps = [c for c in components if isinstance(c, LinearComponent)]
    liesel_comps = [c for c in components if isinstance(c, LieselSplineComponent)]
    mgcv_comps = [c for c in components if isinstance(c, MGCVComponent)]

    assert len(linear_comps) == 1
    assert len(liesel_comps) == 1
    assert len(mgcv_comps) == 1

    assert linear_comps[0].depends_on_vars == {"x1"}
    assert liesel_comps[0].depends_on_vars == {"x2"}
    assert mgcv_comps[0].depends_on_vars == {"x3"}


def test_parse_mgcv_function_kwargs(registry):
    parser = FormulaParser(registry)
    components = parser.parse(
        "s(x1, k=10, bs=\"cr\", fx=True, name='custom')",
        default_intercept=False,
        merge=False,
    )

    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, MGCVComponent)
    assert comp.depends_on_vars == {"x1"}
    assert comp.term_name == "custom"
    assert comp.mgcv_kwargs == {"k": 10, "bs": "cr", "fx": True}


def test_parse_liesel_function_kwargs(registry):
    parser = FormulaParser(registry)
    components = parser.parse(
        "ls(x1, k=10, bs='ps', name='custom')", default_intercept=False, merge=False
    )

    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, LieselSplineComponent)
    assert comp.depends_on_vars == {"x1"}
    assert comp.term_name == "custom"
    assert comp.spline_kwargs == {"k": 10, "bs": "ps"}


def test_merge_multiple_linear_variables(registry):
    parser = FormulaParser(registry)
    components = parser.parse("x1 + x2 + x3", default_intercept=False, merge=True)

    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, LinearComponent)
    assert comp.depends_on_vars == {"x1", "x2", "x3"}


def test_intercept_application_to_merged_linear(registry):
    parser = FormulaParser(registry)

    components = parser.parse("0 + x1 + x2", default_intercept=False, merge=True)
    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, LinearComponent)
    assert comp.depends_on_vars == {"x1", "x2"}
    assert comp.includes_intercept is False

    components = parser.parse("1 + x1 + x2", default_intercept=False, merge=True)
    assert len(components) == 1
    comp = components[0]
    assert isinstance(comp, LinearComponent)
    assert comp.depends_on_vars == {"x1", "x2"}
    assert comp.includes_intercept is True


def test_default_intercept_behavior(registry):
    """Test default intercept behavior when no explicit intercept is specified."""
    parser = FormulaParser(registry)

    # With default_intercept=True
    components = parser.parse("x1 + x2", default_intercept=True, merge=True)
    comp = components[0]
    assert isinstance(comp, LinearComponent)
    assert comp.includes_intercept is True  # Should include intercept by default

    # With default_intercept=False
    components = parser.parse("x1 + x2", default_intercept=False, merge=True)
    comp = components[0]
    assert isinstance(comp, LinearComponent)
    assert comp.includes_intercept is False  # Should not include intercept by default


# Error Handling Tests


def test_unknown_function_names(registry):
    parser = FormulaParser(registry)
    with pytest.raises(ValueError):
        parser.parse("unknown(x1)", default_intercept=False, merge=False)


def test_intercept_conflict_detection(registry):
    parser = FormulaParser(registry)
    with pytest.raises(ValueError):
        parser.parse("0 + 1 + x1", default_intercept=False, merge=False)


def test_invalid_syntax(registry):
    parser = FormulaParser(registry)

    # Empty term after +
    with pytest.raises(ValueError):
        parser.parse("x1 + ", default_intercept=False, merge=False)

    # Empty term before +
    with pytest.raises(ValueError):
        parser.parse(" + x1", default_intercept=False, merge=False)

    # Malformed function call
    with pytest.raises(ValueError):
        parser.parse("ls(x1, k=)", default_intercept=False, merge=False)


def test_empty_formulas(registry):
    parser = FormulaParser(registry)

    # Empty formula
    with pytest.raises(ValueError):
        parser.parse("", default_intercept=False, merge=False)

    with pytest.raises(ValueError):
        parser.parse("   ", default_intercept=False, merge=False)


def test_parse_function_with_nested_parentheses(registry):
    """Test parsing function calls with nested parentheses in parameters."""
    parser = FormulaParser(registry)

    # This should not fail due to nested parentheses in string literal
    components = parser.parse(
        "s(x1, name='m(x1)')", default_intercept=False, merge=False
    )

    assert len(components) == 1
    comp = components[0]
    assert hasattr(comp, "term_name") and comp.term_name == "m(x1)"
