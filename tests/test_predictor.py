import jax.numpy as jnp
import liesel.model as lsl

import liesel_gam as gam


class TestPredictor:
    def test_empty(self) -> None:
        pred = gam.AdditivePredictor("loc")
        assert jnp.allclose(pred.value, 0.0)

    def test_add_term(self) -> None:
        pred = gam.AdditivePredictor("loc")
        pred += lsl.Var(1.0)
        assert jnp.allclose(pred.value, 1.0)

        pred += lsl.Var(2.0)
        assert jnp.allclose(pred.value, 3.0)

    def test_inv_link(self) -> None:
        pred = gam.AdditivePredictor("loc", inv_link=jnp.exp)
        pred += lsl.Var(0.0)
        assert jnp.allclose(pred.value, 1.0)

    def test_access_term(self) -> None:
        pred = gam.AdditivePredictor("loc")
        x = lsl.Var(0.0, name="x")
        pred += x
        assert pred["x"] is x
