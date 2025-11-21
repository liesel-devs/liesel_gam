import logging

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
from ryp import r, to_py

import liesel_gam.builder as gb
from liesel_gam.builder.builder import BasisBuilder
from liesel_gam.builder.registry import PandasRegistry

from .make_df import make_test_df


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

    def test_ri_basis(self) -> None:
        data = pd.DataFrame(
            {"x": pd.Categorical(["a", "b"], categories=["a", "b", "c"])}
        )
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.ri("x")
        assert basis.value.size == 2


class TestFoBasisLinearNumeric:
    def test_name(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("y + x_float", name="X")

        assert basis.name == "B(X)"

    def test_removing_intercept_manually_is_forbidden(self, data):
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(ValueError):
            bases.lin("-1 + y + x_float", name="X")

        with pytest.raises(ValueError):
            bases.lin("0 + y + x_float", name="X")

    def test_adding_intercept_manually_is_forbidden(self, data):
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(ValueError):
            bases.lin("1 + y + x_float", name="X")

    def test_removing_intercept_manually_does_not_interfere_with_names(self, data):
        # edge case with unfortunate variable name
        data["-1"] = data["x_float"]
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("y + `-1`", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["-1"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_simple_linear(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("y + x_float", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)


class TestFoBasisOperators:
    def test_string_literal(self, bases) -> None:
        with pytest.raises(FormulaSyntaxError):
            bases.lin("y + 'string_literal'", name="X")

    def test_numeric_literal(self, bases) -> None:
        with pytest.raises(FormulaSyntaxError):
            bases.lin("y + 5", name="X")

    def test_special_names_with_backticks(self, bases) -> None:
        # name with space
        basis = bases.lin("y + `with space`", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        with_space = bases.data["with space"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], with_space)

        # weird name
        basis = bases.lin("y + `weird:col*name`", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        weird_name = bases.data["weird:col*name"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], weird_name)

    def test_python_function(self, bases) -> None:
        def subtract_five(x):
            return x - 5

        basis = bases.lin(
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
        basis = bases.lin("y + {x_float-5}", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy() - 5

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_grouped_operation(self, bases) -> None:
        basis = bases.lin("y + (x_float-1)", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_wildcard(self, bases) -> None:
        with pytest.raises(FormulaParsingError):
            bases.lin(".", name="X")

    def test_nth_order_interactions(self, bases) -> None:
        basis = bases.lin("(y + x_float + x_int)**2", name="X")

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

        basis2 = bases.lin("(y + x_float + x_int)^2", name="X")

        assert jnp.allclose(basis.value, basis2.value)

    def test_simple_linear_with_double_terms(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("y + y + x_float", name="X")

        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_simple_interaction(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("y + x_float + y:x_float", name="X")

        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)
        assert jnp.allclose(basis.value[:, 2], y * x_float)

    def test_interaction(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("y*x_float", name="X")

        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)
        assert jnp.allclose(basis.value[:, 2], y * x_float)

    def test_interaction_with_double_terms(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("y + x_float + y:x_float + y*x_float", name="X")

        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy()

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)
        assert jnp.allclose(basis.value[:, 2], y * x_float)

    def test_nesting(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis1 = bases.lin("y/x_float", name="X")
        basis2 = bases.lin("y + y:x_float", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

        basis1 = bases.lin("(y + x_float) / x_int", name="X")
        basis2 = bases.lin("y + x_float + y:x_float:x_int", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

    def test_inverted_nesting(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis1 = bases.lin("y/x_float", name="X")
        basis2 = bases.lin("x_float %in% y", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

    def test_remove_term(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis1 = bases.lin("y + x_float - x_float", name="X")
        basis2 = bases.lin("y", name="X")

        assert jnp.allclose(basis1.value, basis2.value)

    def test_inert_plus(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(FormulaSyntaxError):
            bases.lin("y +", name="X")

    def test_split_formula(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(FormulaSyntaxError):
            bases.lin(r"y + x_float \| x_int", name="X")

    def test_tilde_in_formula(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        with pytest.raises(ValueError):
            bases.lin("y ~ x_float", name="X")


class TestFoBasisTransforms:
    def test_identity_transform(self, bases) -> None:
        basis = bases.lin("y + I(x_float-5)", name="X")
        assert basis.value.shape == (84, 2)

        y = bases.data["y"].to_numpy()
        x_float = bases.data["x_float"].to_numpy() - 5

        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], x_float)

    def test_lookup_q(self, bases) -> None:
        with pytest.raises(FactorEvaluationError):
            bases.lin("y + Q('with space')", name="X")

    def test_center(self, bases) -> None:
        data = make_test_df(perturb=False)
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = BasisBuilder(registry)
        basis = bases.lin("y + center(x_float)", name="X")
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
        basis = bases.lin("y + scale(x_float)", name="X")
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
        basis = bases.lin("y + standardize(x_float)", name="X")
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
        basis = bases.lin("y + scale(x_float)", name="X")
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
        basis = bases.lin("y + lag(y)", name="X")
        assert basis.value.shape == (83, 2)

        assert jnp.allclose(basis.value[:, 0], bases.data["y"].to_numpy()[1:])
        assert jnp.allclose(basis.value[:, 1], bases.data["y"].to_numpy()[:-1])

        basis = bases.lin("y + lag(y, 2)", name="X")
        assert basis.value.shape == (82, 2)
        assert jnp.allclose(basis.value[:, 0], bases.data["y"].to_numpy()[2:])
        assert jnp.allclose(basis.value[:, 1], bases.data["y"].to_numpy()[:-2])

    def test_exp(self, bases) -> None:
        basis = bases.lin("y + exp(y)", name="X")
        assert basis.value.shape == (84, 2)

        assert jnp.allclose(basis.value[:, 1], np.exp(bases.data["y"].to_numpy()))

    def test_poly(self, bases) -> None:
        basis = bases.lin("poly(y, degree=3, raw=True)", name="X")
        assert basis.value.shape == (84, 3)

        y = bases.data["y"].to_numpy()
        assert jnp.allclose(basis.value[:, 0], y)
        assert jnp.allclose(basis.value[:, 1], y**2)
        assert jnp.allclose(basis.value[:, 2], y**3)

    def test_bs(self, bases) -> None:
        basis = bases.lin("bs(y, df=6, degree=3)", name="X")
        assert basis.value.shape == (84, 6)

    def test_cs(self, bases) -> None:
        basis = bases.lin("cs(y, df=6)", name="X")
        assert basis.value.shape == (84, 6)

    def test_cc(self, bases) -> None:
        basis = bases.lin("cc(y, df=6)", name="X")
        assert basis.value.shape == (84, 6)

    def test_cr(self, bases) -> None:
        basis = bases.lin("cc(y, df=6)", name="X")
        assert basis.value.shape == (84, 6)

    def test_te(self, bases) -> None:
        with pytest.raises(FactorEvaluationError):
            bases.lin("te(y, x_float)", name="X")

    def test_hashed(self, bases) -> None:
        basis = bases.lin("hashed(y, levels=5)", name="X")
        assert basis.value.shape == (84, 5)


class TestFoBasisLinearCategorical:
    def test_simple_categorical_unordered(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("C(cat_unordered)", name="X")

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

    def test_simple_categorical_ordered(self, data) -> None:
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("C(cat_ordered)", name="X")

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

    def test_category_with_unobserved_label(self) -> None:
        data = pd.DataFrame(
            {"x": pd.Categorical(["a", "b"], categories=["a", "b", "c"])}
        )
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)
        basis = bases.lin("C(x)", name="X")
        assert basis.value.shape == (2, 2)

    def test_category_with_unobserved_label_logging(self, caplog) -> None:
        data = pd.DataFrame(
            {"x": pd.Categorical(["a", "b"], categories=["a", "b", "c"])}
        )
        registry = gb.PandasRegistry(data, na_action="drop")
        bases = gb.BasisBuilder(registry)

        with caplog.at_level(logging.INFO, logger="liesel_gam"):
            basis = bases.lin("C(x)", name="X")

        assert basis.value.shape == (2, 2)
        assert "categories without observations" in caplog.text
        assert caplog.records[0].levelno == logging.INFO


@pytest.fixture(scope="module")
def columb():
    r("library(mgcv)")
    r("data(columb)")
    columb = to_py("columb", format="pandas")
    return columb


@pytest.fixture(scope="module")
def columb_polys():
    r("library(mgcv)")
    r("data(columb.polys)")
    polys = to_py("columb.polys", format="numpy")
    # turn to zero-based indecing
    polys = {k: v - 1 for k, v in polys.items()}
    return polys


class TestMRFBasis:
    def test_initialization_nb_strings(self, columb, columb_polys):
        """
        There are quite a few ways to initialize the MRF basis.
        This test ensures that they all run and lead to consistent results, with
        some hand-selected expected exceptions.
        """
        registry = PandasRegistry(columb)
        bases = BasisBuilder(registry)

        columb_polys
        basis, _, neighbors, labels = bases.mrf("district", polys=columb_polys)

        label_arr = np.asarray(list(columb_polys))
        # label_arr = np.asarray(labels)
        l2i = bases.mappings["district"].labels_to_integers
        nb_int = {k: l2i(v) for k, v in neighbors.items()}

        neighbor_labels = {k: label_arr[v] for k, v in nb_int.items()}

        # string numpy array
        basis2, _, neighbors2, labels2 = bases.mrf("district", nb=neighbor_labels)

        # list of strings
        neighbor_labels = {k: v.tolist() for k, v in neighbor_labels.items()}
        basis3, _, neighbors3, labels3 = bases.mrf("district", nb=neighbor_labels)

        # integer numpy array
        basis4, _, neighbors4, labels4 = bases.mrf("district", nb=nb_int)

        # list of integers
        nb_intlist = {k: v.tolist() for k, v in nb_int.items()}
        basis5, _, neighbors5, labels5 = bases.mrf("district", nb=nb_intlist)

        # float numpy array
        nb_float = {k: np.astype(v, float) for k, v in nb_int.items()}
        basis6, _, neighbors6, labels6 = bases.mrf("district", nb=nb_float)

        # list of floats
        nb_floatlist = {k: v.tolist() for k, v in nb_float.items()}
        basis7, _, neighbors7, labels7 = bases.mrf("district", nb=nb_floatlist)

        for b in [basis2, basis3, basis4, basis5, basis6, basis7]:
            assert jnp.allclose(b.value, basis.value)
            assert jnp.allclose(b.penalty.value, basis.penalty.value)

        nb_list = [
            neighbors2,
            neighbors3,
            neighbors4,
            neighbors5,
            neighbors6,
            neighbors7,
        ]
        for nb in nb_list:
            for key, nb_list in nb.items():
                assert nb_list == neighbors[key]

        for lab in [
            labels2,
            labels3,
            labels4,
            labels5,
            labels6,
            labels7,
        ]:
            assert lab == labels

    def test_initialization_consistency(self, columb, columb_polys):
        """
        There are quite a few ways to initialize the MRF basis.
        This test ensures that they all run and lead to consistent results, with
        some hand-selected expected exceptions.
        """
        registry = PandasRegistry(columb)
        bases = BasisBuilder(registry)

        basis, _, neighbors, labels = bases.mrf("district", polys=columb_polys)

        basis2, _, neighbors2, labels2 = bases.mrf("district", nb=neighbors)

        basis3, _, neighbors3, labels3 = bases.mrf(
            "district", penalty=basis.penalty.value
        )

        basis4, _, neighbors4, labels4 = bases.mrf(
            "district", polys=columb_polys, nb=neighbors
        )

        basis5, _, neighbors5, labels5 = bases.mrf(
            "district", polys=columb_polys, penalty=basis.penalty.value
        )

        basis6, _, neighbors6, labels6 = bases.mrf(
            "district", nb=neighbors, penalty=basis.penalty.value
        )

        basis7, _, neighbors7, labels7 = bases.mrf(
            "district", polys=columb_polys, nb=neighbors, penalty=basis.penalty.value
        )

        for b in [basis2, basis3, basis4, basis5, basis6, basis7]:
            assert jnp.allclose(b.value, basis.value)
            assert jnp.allclose(b.penalty.value, basis.penalty.value)

        nb_list = [
            neighbors2,
            neighbors3,
            neighbors4,
            neighbors5,
            neighbors6,
            neighbors7,
        ]
        for i, nb in enumerate(nb_list):
            # nb_list[0]  # not none (has neighbors)
            # nb_list[1]  # none (only penalty)
            # nb_list[2]  # not none (polys and neighbors)
            # nb_list[3]  # none (polys and penalty)
            # nb_list[4]  # not none (neighbors and penalty)
            # nb_list[5]  # not none (polys, neighbors and penalty)

            if i in [1, 3]:
                # because in these cases, the smooth does not compute the neighbor list
                assert nb is None
            else:
                for key, nb_list in nb.items():
                    assert nb_list == neighbors[key]

        for lab in [
            labels2,
            labels3,
            labels4,
            labels5,
            labels6,
            labels7,
        ]:
            assert lab == labels

    def test_initialization_consistency_low_rank(self, columb, columb_polys):
        """
        Low-rank approximations only work if the penalty is *not* supplied.
        """
        registry = PandasRegistry(columb)
        bases = BasisBuilder(registry)

        basis, _, neighbors, labels = bases.mrf("district", k=20, polys=columb_polys)

        basis2, _, neighbors2, labels2 = bases.mrf("district", k=20, nb=neighbors)

        with pytest.raises(ValueError):
            bases.mrf("district", k=20, penalty=basis.penalty.value)

        basis4, _, neighbors4, labels4 = bases.mrf(
            "district", k=20, polys=columb_polys, nb=neighbors
        )

        with pytest.raises(ValueError):
            bases.mrf("district", k=20, polys=columb_polys, penalty=basis.penalty.value)

        with pytest.raises(ValueError):
            bases.mrf("district", k=20, nb=neighbors, penalty=basis.penalty.value)

        with pytest.raises(ValueError):
            bases.mrf(
                "district",
                k=20,
                polys=columb_polys,
                nb=neighbors,
                penalty=basis.penalty.value,
            )

        for b in [basis2, basis4]:
            assert jnp.allclose(b.value, basis.value)
            assert jnp.allclose(b.penalty.value, basis.penalty.value)

        nb_list = [
            neighbors2,
            neighbors4,
        ]
        for i, nb in enumerate(nb_list):
            for key, nb_list in nb.items():
                assert nb_list == neighbors[key]

        for lab in [
            labels2,
            labels4,
        ]:
            assert lab == labels
