import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
from formulaic.errors import (
    FactorEvaluationError,
    FormulaParsingError,
    FormulaSyntaxError,
)

import liesel_gam.builder as gb
from liesel_gam.builder.builder import BasisBuilder
from liesel_gam.testing_df import make_test_df


@pytest.fixture(scope="module")
def data():
    return make_test_df()


@pytest.fixture(scope="class")
def bases(data) -> gb.BasisBuilder:
    registry = gb.PandasRegistry(data, na_action="drop")
    bases = gb.BasisBuilder(registry)
    return bases


class TestBasisBuilder:
    def test_init(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        gb.BasisBuilder(registry)


class TestFoBasisLinearNumeric:
    def test_name(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("y + x_float", name="X")

        assert basis.name == "B(X)"

    def test_removing_intercept_manually_is_forbidden(self, data):
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(RuntimeError):
            bases.fo("-1 + y + x_float", name="X")

        with pytest.raises(RuntimeError):
            bases.fo("0 + y + x_float", name="X")

    def test_removing_intercept_manually_does_not_interfere_with_names(self, data):
        # edge case with unfortunate variable name
        data["-1"] = data["x_float"]
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("y + `-1`", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["-1"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_simple_linear(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("y + x_float", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    @pytest.mark.xfail(reason="currently broken")
    def test_special_names_with_q(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        # Alernative method
        basis = bases.fo("y + Q('weird:col*name')", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        weird_name = bases.data["weird:col*name"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], weird_name)

    @pytest.mark.xfail(reason="Currently broken")
    def test_explicit_removed_intercept_m1(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("-1 + y + x_float", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    @pytest.mark.xfail(reason="Currently broken")
    def test_explicit_removed_intercept_0(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("0 + y + x_float", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)


class TestFoBasisOperators:
    def test_string_literal(self, bases) -> None:
        with pytest.raises(FormulaSyntaxError):
            bases.fo_basis("y + 'string_literal'", name="X")

    def test_numeric_literal(self, bases) -> None:
        with pytest.raises(FormulaSyntaxError):
            bases.fo_basis("y + 5", name="X")

    def test_special_names_with_backticks(self, bases) -> None:
        # name with space
        basis = bases.fo_basis("y + `with space`", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        with_space = bases.data["with space"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], with_space)

        # weird name
        basis = bases.fo_basis("y + `weird:col*name`", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        weird_name = bases.data["weird:col*name"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], weird_name)

    def test_python_function(self, bases) -> None:
        def subtract_five(x):
            return x - 5

        basis = bases.fo_basis(
            "y + subtract_five(x_float)",
            name="X",
            context={"subtract_five": subtract_five},
        )

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy() - 5

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_quoted_python(self, bases) -> None:
        basis = bases.fo_basis("y + {x_float-5}", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy() - 5

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_grouped_operation(self, bases) -> None:
        basis = bases.fo_basis("y + (x_float-1)", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_wildcard(self, bases) -> None:
        with pytest.raises(FormulaParsingError):
            bases.fo_basis(".", name="X")

    def test_nth_order_interactions(self, bases) -> None:
        basis = bases.fo_basis("(y + x_float + x_int)**2", name="X")

        assert basis.value.shape == (84, 6)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()
        x_int = bases.data["x_int"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)
        assert jnp.allclose(basis.value[:, 2], x_int)
        assert jnp.allclose(basis.value[:, 3], y * x_float)
        assert jnp.allclose(basis.value[:, 4], y * x_int)
        assert jnp.allclose(basis.value[:, 5], x_float * x_int)

        basis2 = bases.fo_basis("(y + x_float + x_int)^2", name="X")

        assert jnp.allclose(basis.value, basis2.value)

    def test_simple_linear_with_double_terms(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("y + y + x_float", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_simple_interaction(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("y + x_float + y:x_float", name="X")

        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)
        assert jnp.allclose(basis.value[:, 2], y * x_float)

    def test_interaction(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("y*x_float", name="X")

        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)
        assert jnp.allclose(basis.value[:, 2], y * x_float)

    def test_interaction_with_double_terms(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("y + x_float + y:x_float + y*x_float", name="X")

        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)
        assert jnp.allclose(basis.value[:, 2], y * x_float)

    def test_nesting(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis1 = bases.fo("y/x_float", name="X")
        basis2 = bases.fo("y + y:x_float", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

        basis1 = bases.fo("(y + x_float) / x_int", name="X")
        basis2 = bases.fo("y + x_float + y:x_float:x_int", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

    def test_inverted_nesting(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis1 = bases.fo("y/x_float", name="X")
        basis2 = bases.fo("x_float %in% y", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

    def test_remove_term(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis1 = bases.fo("y + x_float - x_float", name="X")
        basis2 = bases.fo("y", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

    def test_inert_plus(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(FormulaSyntaxError):
            bases.fo("y +", name="X")

    def test_split_formula(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(FormulaSyntaxError):
            bases.fo(r"y + x_float \| x_int", name="X")

    def test_tilde_in_formula(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(ValueError):
            bases.fo("y ~ x_float", name="X")


class TestFoBasisTransforms:
    def test_identity_transform(self, bases) -> None:
        basis = bases.fo_basis("y + I(x_float-5)", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy() - 5

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_lookup_q(self, bases) -> None:
        with pytest.raises(FactorEvaluationError):
            bases.fo_basis("y + Q('with space')", name="X")

    def test_center(self, bases) -> None:
        data = make_test_df(perturb=False)
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = BasisBuilder(registry)
        basis = bases.fo("y + center(x_float)", name="X")
        assert basis.value.shape == (89, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()
        x_float = x_float - x_float.mean()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float, rtol=1e-4)
        assert basis.value[:, 1].mean() == pytest.approx(0.0, abs=1e-6)

    def test_scale(self, bases) -> None:
        data = make_test_df(perturb=False)
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = BasisBuilder(registry)
        basis = bases.fo("y + scale(x_float)", name="X")
        assert basis.value.shape == (89, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()
        x_float = (x_float - x_float.mean()) / x_float.std()

        assert jnp.allclose(basis.value[:, 0], y)
        assert ((basis.value[:, 1] - x_float) ** 2).sum() < 0.003
        assert basis.value[:, 1].mean() == pytest.approx(0.0, abs=1e-6)
        assert basis.value[:, 1].std() == pytest.approx(1.0, abs=1e-2)

    def test_standardize(self, bases) -> None:
        data = make_test_df(perturb=False)
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = BasisBuilder(registry)
        basis = bases.fo("y + standardize(x_float)", name="X")
        assert basis.value.shape == (89, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()
        x_float = (x_float - x_float.mean()) / x_float.std()

        assert jnp.allclose(basis.value[:, 0], y)
        assert ((basis.value[:, 1] - x_float) ** 2).sum() < 0.003
        assert basis.value[:, 1].mean() == pytest.approx(0.0, abs=1e-6)
        assert basis.value[:, 1].std() == pytest.approx(1.0, abs=1e-2)

    def test_predict_with_scale(self) -> None:
        data = make_test_df(perturb=False)
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = BasisBuilder(registry)
        basis = bases.fo("y + scale(x_float)", name="X")
        assert basis.value.shape == (89, 2)

        x_float = bases.data["x_float"].to_numpy()
        x_mean = x_float.mean()
        x_std = x_float.std()

        new = np.linspace(0, 5, 30)

        model = lsl.Model([basis])
        model.update_state({"y": new, "x_float": new}, inplace=True)

        assert basis.value.shape == (30, 2)

        assert jnp.allclose(basis.value[:, 1], (new - x_mean) / x_std, atol=1e-1)
        squared_error = ((basis.value[:, 1] - ((new - x_mean) / x_std)) ** 2).sum()
        assert squared_error < 0.02

    def test_lag(self, bases) -> None:
        basis = bases.fo_basis("y + lag(y)", name="X")
        assert basis.value.shape == (83, 2)

        assert jnp.allclose(basis.value[:, 0], bases.data["y"].to_numpy()[1:])
        assert jnp.allclose(basis.value[:, 1], bases.data["y"].to_numpy()[:-1])

        basis = bases.fo_basis("y + lag(y, 2)", name="X")
        assert basis.value.shape == (82, 2)
        assert jnp.allclose(basis.value[:, 0], bases.data["y"].to_numpy()[2:])
        assert jnp.allclose(basis.value[:, 1], bases.data["y"].to_numpy()[:-2])

    def test_exp(self, bases) -> None:
        basis = bases.fo_basis("y + exp(y)", name="X")
        assert basis.value.shape == (84, 2)

        assert jnp.allclose(basis.value[:, 1], np.exp(bases.data["y"].to_numpy()))

    def test_poly(self, bases) -> None:
        basis = bases.fo_basis("poly(y, degree=3, raw=True)", name="X")
        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], y**2)
        assert jnp.allclose(basis.value[:, 2], y**3)

    def test_bs(self, bases) -> None:
        basis = bases.fo_basis("bs(y, df=6, degree=3)", name="X")
        assert basis.value.shape == (84, 6)

    def test_cs(self, bases) -> None:
        basis = bases.fo_basis("cs(y, df=6)", name="X")
        assert basis.value.shape == (84, 6)

    def test_cc(self, bases) -> None:
        basis = bases.fo_basis("cc(y, df=6)", name="X")
        assert basis.value.shape == (84, 6)

    def test_cr(self, bases) -> None:
        basis = bases.fo_basis("cc(y, df=6)", name="X")
        assert basis.value.shape == (84, 6)

    def test_te(self, bases) -> None:
        with pytest.raises(FactorEvaluationError):
            bases.fo_basis("te(y, x_float)", name="X")

    def test_hashed(self, bases) -> None:
        basis = bases.fo_basis("hashed(y, levels=5)", name="X")
        assert basis.value.shape == (84, 5)


class TestFoBasisLinearCategorical:
    def test_simple_categorical_unordered(self, data) -> data:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("C(cat_unordered)", name="X")

        mapping = bases.mappings["cat_unordered"].labels_to_integers_map

        ncat = len(mapping)
        assert basis.value.shape == (84, ncat - 1)

        pd_codes = pd.Categorical(bases.data["cat_unordered"]).codes
        bases_codes = bases.mappings["cat_unordered"].labels_to_integers(
            bases.data["cat_unordered"]
        )

        assert np.all(pd_codes == bases_codes)

        bool_cat2 = np.asarray(basis.value[:, 0] == 1)
        cat2 = bases.data[bool_cat2]["cat_unordered"].to_numpy()
        assert np.all(cat2 == "B")

        bool_cat3 = np.asarray(basis.value[:, 1] == 1)
        cat3 = bases.data[bool_cat3]["cat_unordered"].to_numpy()
        assert np.all(cat3 == "C")

    def test_simple_categorical_ordered(self, data) -> data:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.fo("C(cat_ordered)", name="X")

        mapping = bases.mappings["cat_ordered"].labels_to_integers_map

        ncat = len(mapping)
        assert basis.value.shape == (84, ncat - 1)

        pd_codes = pd.Categorical(bases.data["cat_ordered"]).codes
        bases_codes = bases.mappings["cat_ordered"].labels_to_integers(
            bases.data["cat_ordered"]
        )

        assert np.all(pd_codes == bases_codes)

        bool_cat2 = np.asarray(basis.value[:, 0] == 1)
        cat2 = bases.data[bool_cat2]["cat_ordered"].to_numpy()
        assert np.all(cat2 == "med")

        bool_cat3 = np.asarray(basis.value[:, 1] == 1)
        cat3 = bases.data[bool_cat3]["cat_ordered"].to_numpy()
        assert np.all(cat3 == "high")
