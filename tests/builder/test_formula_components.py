"""Tests for formula components."""

import pandas as pd
import pytest

from liesel_gam.builder.components import FactorComponent
from liesel_gam.builder.formula import FormulaParser
from liesel_gam.builder.registry import VariableRegistry


class TestFactorComponent:
    @pytest.fixture
    def sample_data(self):
        # create test data with 3 levels
        data = pd.DataFrame(
            {"category": pd.Categorical(["A", "B", "C", "A", "B", "C", "A"])}
        )
        return data

    def test_factor_component_basic(self):
        # create factor component
        factor_comp = FactorComponent("category")

        # test formula key
        assert factor_comp.formula_key(simplified=True) == "factor(category)"

        # test properties
        assert not factor_comp.includes_intercept
        assert factor_comp.depends_on_vars == {"category"}

    def test_factor_component_treatment_encoding(self, sample_data):
        registry = VariableRegistry(sample_data)

        # create factor component
        factor_comp = FactorComponent("category")

        # create terms
        terms = factor_comp.create_terms(registry)

        # should create one LinearTerm
        assert len(terms) == 1
        linear_term = terms[0]

        # check the dummy matrix shape and values
        dummy_matrix = linear_term.basis.value
        assert dummy_matrix.shape == (7, 2)  # n_obs=7, n_categories-1=2 (B and C)

        # check dummy encoding is correct
        # A=reference (all zeros), B=[1,0], C=[0,1]
        expected = [
            [0, 0],  # A
            [1, 0],  # B
            [0, 1],  # C
            [0, 0],  # A
            [1, 0],  # B
            [0, 1],  # C
            [0, 0],  # A
        ]

        for i, row in enumerate(expected):
            assert dummy_matrix[i, 0] == row[0], (
                f"Row {i}, col 0: expected {row[0]}, got {dummy_matrix[i, 0]}"
            )
            assert dummy_matrix[i, 1] == row[1], (
                f"Row {i}, col 1: expected {row[1]}, got {dummy_matrix[i, 1]}"
            )

    def test_factor_component_from_formula(self, sample_data):
        registry = VariableRegistry(sample_data)
        parser = FormulaParser(registry)
        components = parser.parse(
            "factor(category)", default_intercept=False, merge=False
        )

        assert len(components) == 1
        comp = components[0]

        # test basic creation
        assert isinstance(comp, FactorComponent)
        assert comp.var_name == "category"
        assert comp.contrasts == "treatment"
        assert comp.reference_level is None
        assert comp._name is None
