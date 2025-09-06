import jax.numpy as jnp
import pytest

import liesel_gam as gam

term1 = gam.SmoothTerm.new_ig(
    basis=gam.Basis.new_linear(1.0, xname="x1"), penalty=jnp.eye(1), name="s(x1)"
)

term1.coef.value = 1.0
term1.update()

term2 = gam.SmoothTerm.new_ig(
    basis=gam.Basis.new_linear(1.0, xname="x1"), penalty=jnp.eye(1), name="s(x2)"
)

term2.coef.value = 2.0
term2.update()


class TestPredictor:
    def test_empty(self) -> None:
        pred = gam.AdditivePredictor("loc")
        assert jnp.allclose(pred.value, 0.0)

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
        assert len(pred.terms) == 2

        # list
        pred = gam.AdditivePredictor("loc")
        pred += [term1, term2]
        assert jnp.allclose(pred.value, 3.0)
        assert len(pred.terms) == 2

    def test_add_term_with_same_name(self) -> None:
        pred = gam.AdditivePredictor("loc")
        with pytest.raises(RuntimeError):
            pred += term1, term1

    def test_inv_link(self) -> None:
        pred = gam.AdditivePredictor("loc", inv_link=jnp.exp)
        pred += term1
        assert jnp.allclose(pred.value, jnp.exp(1.0))

    def test_access_term(self) -> None:
        pred = gam.AdditivePredictor("loc")
        pred += term1
        assert pred[term1.name] is term1

    def test_intercept_none(self) -> None:
        pred = gam.AdditivePredictor("loc")
        term = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.atleast_2d(1.0), lambda x: x, xname="x1"),
            penalty=jnp.eye(1),
            name="s(x1)",
        )
        pred += term

        assert pred.includes_intercept is None

        # when a second term is added that has no intercept,
        # we still don't know whether the first term might have one.
        term2 = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.atleast_2d(1.0), xname="x2", includes_intercept=False),
            penalty=jnp.eye(1),
            name="s(x2)",
        )
        pred += term2
        assert pred.includes_intercept is None

    def test_intercept_true(self) -> None:
        pred = gam.AdditivePredictor("loc")
        term = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.atleast_2d(1.0), xname="x1"),
            penalty=jnp.eye(1),
            name="s(x1)",
        )
        pred += term
        assert pred.includes_intercept is None

        # when a second term is added that has an intercept,
        # we know the predictor contains an intercept
        term2 = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.atleast_2d(1.0), xname="x2", includes_intercept=True),
            penalty=jnp.eye(1),
            name="s(x2)",
        )
        pred += term2
        assert pred.includes_intercept is True

    def test_intercept_false(self) -> None:
        pred = gam.AdditivePredictor("loc")
        term = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.atleast_2d(1.0), xname="x1", includes_intercept=False),
            penalty=jnp.eye(1),
            name="s(x1)",
        )
        pred += term

        assert pred.includes_intercept is False
