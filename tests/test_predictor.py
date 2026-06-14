import jax.numpy as jnp
import liesel.model as lsl
import pytest

import liesel_gam as gam

term1 = gam.SmoothTerm.f(
    basis=gam.Basis.new_linear(jnp.array(1.0), xname="x1"),
    scale=gam.VarIGPrior(1.0, 0.005),
    fname="s1",
)

term1.coef.value = jnp.array(1.0)
term1.update()

term2 = gam.SmoothTerm.f(
    basis=gam.Basis.new_linear(jnp.array(1.0), xname="x1"),
    scale=gam.VarIGPrior(1.0, 0.005),
    fname="s2",
)

term2.coef.value = jnp.array(2.0)
term2.update()


class TestPredictor:
    def test_empty(self) -> None:
        pred = gam.AdditivePredictor("loc")
        assert jnp.allclose(pred.value, 0.0)

    def test_add_intercept(self) -> None:
        a = lsl.Var(1.0)
        pred = gam.AdditivePredictor("loc", intercept=a)
        assert pred.intercept is a

    def test_add_term(self) -> None:
        pred = gam.AdditivePredictor("loc")
        pred += term1
        assert jnp.allclose(pred.value, 1.0)

        pred += term2
        assert jnp.allclose(pred.value, 3.0)

    def test_add_iterables(self) -> None:
        # tuple
        pred = gam.AdditivePredictor("loc")
        pred += term1, term2
        assert jnp.allclose(pred.value, 3.0)
        assert len(pred.terms) == 2  # because intercept is not in terms

        # list
        pred = gam.AdditivePredictor("loc")
        pred += [term1, term2]
        assert jnp.allclose(pred.value, 3.0)
        assert len(pred.terms) == 2  # because intercept is not in terms

    def test_add_term_with_same_name(self) -> None:
        pred = gam.AdditivePredictor("loc")
        with pytest.raises(RuntimeError):
            pred += term1, term1

    def test_inv_link(self) -> None:
        pred = gam.AdditivePredictor("loc", inv_link=jnp.exp)
        pred += term1
        assert jnp.allclose(pred.linear_predictor.value, 1.0)
        assert jnp.allclose(pred.value, jnp.exp(1.0))

    def test_linear_predictor_without_link(self) -> None:
        pred = gam.AdditivePredictor("loc")
        pred += term1

        assert jnp.allclose(pred.linear_predictor.value, 1.0)
        assert jnp.allclose(pred.value, pred.linear_predictor.value)

    def test_linear_predictor_reflects_intercept_updates(self) -> None:
        pred = gam.AdditivePredictor("scale", inv_link=jnp.exp)
        pred += lsl.Var.new_value(1.0, name="s(x)")
        intercept = pred.intercept
        assert isinstance(intercept, lsl.Var)
        intercept.value = 2.0
        pred.update()

        assert jnp.allclose(pred.linear_predictor.value, 3.0)
        assert jnp.allclose(pred.value, jnp.exp(3.0))

    def test_update_refreshes_linear_and_transformed_predictors(self) -> None:
        term = lsl.Var.new_value(1.0, name="s(x)")
        pred = gam.AdditivePredictor("scale", inv_link=jnp.exp, intercept=False)
        pred += term

        term.value = 2.0
        pred.update()

        assert jnp.allclose(pred.linear_predictor.value, 2.0)
        assert jnp.allclose(pred.value, jnp.exp(2.0))

    def test_terms_exclude_intercept_and_linear_predictor(self) -> None:
        pred = gam.AdditivePredictor("loc")
        pred += term1, term2

        assert len(pred.terms) == 2
        assert pred.intercept.name not in pred.terms
        assert pred.linear_predictor.name not in pred.terms

    def test_linear_predictor_default_names(self) -> None:
        pred = gam.AdditivePredictor("loc")
        assert pred.linear_predictor.name == "$\\eta_{loc}$"

        pred = gam.AdditivePredictor("$\\sigma$")
        assert pred.linear_predictor.name == "$\\eta_{\\sigma}$"

    def test_linear_predictor_custom_name(self) -> None:
        pred = gam.AdditivePredictor(
            "loc",
            linear_predictor_name="eta_custom{subscript}",
        )

        assert pred.linear_predictor.name == "eta_custom_{loc}"

    def test_linear_predictor_is_model_node_not_var(self) -> None:
        pred = gam.AdditivePredictor("loc")
        model = lsl.Model([pred])

        assert pred.linear_predictor.name in model.nodes
        assert pred.linear_predictor.name not in model.vars
        assert pred.name in model.vars

    def test_access_term(self) -> None:
        pred = gam.AdditivePredictor("loc")
        pred += term1
        assert pred[term1.name] is term1

    def test_intercept(self) -> None:
        pred = gam.AdditivePredictor("loc")
        assert pred.intercept.name == "$\\beta_{0,loc}$"
        assert pred.intercept is not None
        assert pred.intercept.parameter  # type: ignore

        pred = gam.AdditivePredictor("loc", intercept=False)
        assert isinstance(pred.intercept, lsl.Value)

    def test_repr_default_intercept(self) -> None:
        pred = gam.AdditivePredictor("loc")
        assert (
            repr(pred) == "AdditivePredictor(name='loc', 0 terms, "
            "intercept='$\\\\beta_{0,loc}$')"
        )

    def test_repr_custom_intercept(self) -> None:
        intercept = lsl.Var.new_param(
            value=1.0,
            distribution=None,
            name="b0",
        )
        pred = gam.AdditivePredictor("loc", intercept=intercept)
        assert repr(pred) == "AdditivePredictor(name='loc', 0 terms, intercept='b0')"

    def test_repr_disabled_intercept(self) -> None:
        pred = gam.AdditivePredictor("loc", intercept=False)
        assert repr(pred) == "AdditivePredictor(name='loc', 0 terms, intercept=False)"

    def test_repr_term_count_excludes_intercept(self) -> None:
        pred = gam.AdditivePredictor("loc")
        pred += term1, term2

        assert len(pred.terms) == 2
        assert (
            repr(pred) == "AdditivePredictor(name='loc', 2 terms, "
            "intercept='$\\\\beta_{0,loc}$')"
        )
