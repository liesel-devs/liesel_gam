import jax.numpy as jnp
import liesel.model as lsl

import liesel_gam as gam
from liesel_gam.consolidate_bases import consolidate_bases, evaluate_bases


class TestConsolidateBases:
    def test_consolidate_with_copy(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x1")
        basis1 = gam.Basis(x, basis_fn=lambda x: x**2)

        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x2")
        basis2 = gam.Basis(x, basis_fn=lambda x: x**2)

        pred = lsl.Var.new_calc(lambda b1, b2: b1 + b2, basis1, basis2, name="pred")

        model = lsl.Model([pred])
        assert "x1" in model.vars
        assert "x2" in model.vars

        model_with_strong_bases, model_for_bases = consolidate_bases(model, copy=True)

        assert "x1" not in model_with_strong_bases.vars
        assert "x2" not in model_with_strong_bases.vars
        assert model_with_strong_bases.vars["B(x1)"].strong
        assert model_with_strong_bases.vars["B(x2)"].strong

        assert "x1" in model_for_bases.vars
        assert "x2" in model_for_bases.vars
        assert model_for_bases.vars["B(x1)"].weak
        assert model_for_bases.vars["B(x2)"].weak

        assert model_for_bases.vars["B(x1)"] is not basis1
        assert model_for_bases.vars["B(x2)"] is not basis2

    def test_consolidate_without_copy(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x1")
        basis1 = gam.Basis(x, basis_fn=lambda x: x**2)

        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x2")
        basis2 = gam.Basis(x, basis_fn=lambda x: x**2)

        pred = lsl.Var.new_calc(lambda b1, b2: b1 + b2, basis1, basis2, name="pred")

        model = lsl.Model([pred])
        assert "x1" in model.vars
        assert "x2" in model.vars

        model_with_strong_bases, model_for_bases = consolidate_bases(model, copy=False)

        assert "x1" not in model_with_strong_bases.vars
        assert "x2" not in model_with_strong_bases.vars
        assert model_with_strong_bases.vars["B(x1)"].strong
        assert model_with_strong_bases.vars["B(x2)"].strong

        assert "x1" in model_for_bases.vars
        assert "x2" in model_for_bases.vars
        assert model_for_bases.vars["B(x1)"].weak
        assert model_for_bases.vars["B(x2)"].weak

        assert model_for_bases.vars["B(x1)"] is basis1
        assert model_for_bases.vars["B(x2)"] is basis2

    def test_evaluate_bases(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x1")
        basis1 = gam.Basis(x, basis_fn=lambda x: x**2)

        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x2")
        basis2 = gam.Basis(x, basis_fn=lambda x: x**2)

        pred = lsl.Var.new_calc(lambda b1, b2: b1 + b2, basis1, basis2, name="pred")

        model = lsl.Model([pred])
        assert "x1" in model.vars
        assert "x2" in model.vars

        _, model_for_bases = consolidate_bases(model, copy=True)

        newdata = {"x1": jnp.linspace(0, 1, 5), "x2": jnp.linspace(0, 1, 5)}
        newdata = evaluate_bases(newdata, model_for_bases)
        assert "B(x1)" in newdata
        assert "B(x2)" in newdata

        newdata = {"x1": jnp.linspace(0, 1, 5)}
        newdata = evaluate_bases(newdata, model_for_bases)
        assert "B(x1)" in newdata
        assert "B(x2)" in newdata

        newdata = evaluate_bases({}, model_for_bases)
        assert "B(x1)" in newdata
        assert "B(x2)" in newdata
