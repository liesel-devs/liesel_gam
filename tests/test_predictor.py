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
        assert len(pred.terms) == 3  # because intercept is in there, too

        # list
        pred = gam.AdditivePredictor("loc")
        pred += [term1, term2]
        assert jnp.allclose(pred.value, 3.0)
        assert len(pred.terms) == 3  # because intercept is in there, too

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

    def test_intercept(self) -> None:
        pred = gam.AdditivePredictor("loc")
        assert "loc_intercept" in pred.terms

        pred = gam.AdditivePredictor("loc", add_intercept=False)
        assert "loc_intercept" not in pred.terms

        pred = gam.AdditivePredictor("loc", add_intercept=False)
        pred += gam.Intercept("b0")
        assert "b0" in pred.terms
