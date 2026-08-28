import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pytest

import liesel_gam as gam
from liesel_gam.consolidate_bases import consolidate_bases


class TestMVAdditivePredictor:
    def test_penalty_validation(self) -> None:
        with pytest.raises(ValueError, match="matrix"):
            gam.MVAdditivePredictor("delta", jnp.ones(3))
        with pytest.raises(ValueError, match="square"):
            gam.MVAdditivePredictor("delta", jnp.ones((2, 3)))
        with pytest.raises(ValueError, match="symmetric"):
            gam.MVAdditivePredictor("delta", jnp.array([[1.0, 1.0], [0.0, 1.0]]))
        with pytest.raises(ValueError, match="positive semidefinite"):
            gam.MVAdditivePredictor("delta", jnp.diag(jnp.array([1.0, -1.0])))
        with pytest.raises(ValueError, match="finite"):
            gam.MVAdditivePredictor(
                "delta", jnp.array([[1.0, jnp.nan], [jnp.nan, 1.0]])
            )

    def test_random_walk_constructor_and_scaling(self) -> None:
        predictor = gam.MVAdditivePredictor.from_random_walk(
            "delta", ndim=4, scale_penalty=False
        )
        difference = jnp.diff(jnp.eye(4), axis=0)
        expected = difference.T @ difference

        assert isinstance(predictor.penalty, lsl.Value)
        assert jnp.allclose(predictor.penalty.value, expected)
        assert predictor.ndim == 4
        assert predictor.latent_ndim == 4

        scaled = gam.MVAdditivePredictor.from_random_walk("scaled", ndim=4)
        assert jnp.linalg.norm(scaled.penalty.value, ord=jnp.inf) == pytest.approx(1.0)

    def test_constructor_accepts_numpy_integer_dimension(self) -> None:
        predictor = gam.MVAdditivePredictor.from_random_walk("delta", ndim=np.int64(4))

        assert predictor.ndim == 4

    def test_identity_and_no_penalty_constructors(self) -> None:
        identity = gam.MVAdditivePredictor.from_identity("identity", ndim=3)
        assert jnp.allclose(identity.penalty.value, jnp.eye(3))

        unpenalized = gam.MVAdditivePredictor.from_no_penalty("none", ndim=3)
        assert jnp.allclose(unpenalized.penalty.value, jnp.zeros((3, 3)))
        assert unpenalized.intercept.dist_node is None
        model = lsl.Model([identity])
        assert identity.intercept.name in model.vars

        with pytest.raises(ValueError, match="not identified"):
            gam.MVAdditivePredictor.from_no_penalty(
                "none2", ndim=3, intercept_scale=2.0
            )

    @pytest.mark.parametrize("ndim", (0, -1, 1.5))
    def test_invalid_constructor_dimension(self, ndim) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            gam.MVAdditivePredictor.from_identity("delta", ndim=ndim)

    def test_sumzero_constraint_reconstructs_full_dimension(self) -> None:
        predictor = gam.MVAdditivePredictor.from_random_walk(
            "delta", ndim=4, scale_penalty=False
        ).constrain("sumzero_coef")

        assert predictor.constraint == "sumzero_coef"
        assert predictor.penalty.value.shape == (3, 3)
        assert predictor.dimension_reparam.value.shape == (4, 3)
        assert predictor.latent_ndim == 3
        assert predictor.ndim == 4

        intercept = predictor.intercept
        assert isinstance(intercept, gam.MultivariateIntercept)
        intercept.coef.value = jnp.array([1.0, -0.5, 2.0])
        intercept.update()
        predictor.update()
        assert intercept.value.shape == (4,)
        assert jnp.sum(intercept.value) == pytest.approx(0.0, abs=1e-6)
        assert predictor.value.shape == (4,)

        assert isinstance(intercept.scale, lsl.Var)
        variance = intercept.scale.value_node[0]
        assert isinstance(variance, lsl.Var)
        assert isinstance(variance.inference, gs.MCMCSpec)
        assert variance.inference.kernel_kwargs["coef"] is intercept.coef
        assert jnp.shape(variance.inference.kernel_kwargs["penalty"]) == (3, 3)

    def test_custom_constraint(self) -> None:
        predictor = gam.MVAdditivePredictor.from_identity(
            "delta", ndim=3, intercept=False
        )
        constraint = jnp.array([[1.0, -1.0, 0.0]])
        predictor.constrain(constraint)

        assert predictor.constraint == "custom"
        assert predictor.dimension_reparam.value.shape == (3, 2)
        assert jnp.allclose(constraint @ predictor.dimension_reparam.value, 0.0)

    def test_constraint_guards(self) -> None:
        with pytest.raises(ValueError, match="Unknown constraint"):
            gam.MVAdditivePredictor.from_identity("delta", 3).constrain("unknown")
        with pytest.raises(ValueError, match="at least two"):
            gam.MVAdditivePredictor.from_identity("delta1", 1).constrain("sumzero_coef")

        with pytest.raises(ValueError, match="full row rank"):
            gam.MVAdditivePredictor.from_identity("delta", 3).constrain(
                jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            )

        predictor = gam.MVAdditivePredictor.from_identity(
            "delta", ndim=3, intercept=False
        )
        predictor += lsl.Var.new_value(jnp.ones((5, 3)), name="term")
        with pytest.raises(RuntimeError, match="before terms"):
            predictor.constrain("sumzero_coef")
        with pytest.raises(RuntimeError, match="after terms"):
            predictor.scale_penalty()

    def test_linked_builder_locks_constraint_after_term_construction(self) -> None:
        predictor = gam.MVAdditivePredictor.from_identity(
            "delta", ndim=3, intercept=False
        )
        builder = gam.MVTermBuilder.from_predictor(
            predictor,
            gam.TermBuilder.from_dict({"x": jnp.arange(6.0)}),
        )
        builder.lin("x", dimension_scale=1.0)

        with pytest.raises(RuntimeError, match="before terms"):
            predictor.constrain("sumzero_coef")

    def test_term_dimension_and_structure_guards(self) -> None:
        predictor = gam.MVAdditivePredictor.from_identity(
            "delta", ndim=3, intercept=False
        )
        with pytest.raises(ValueError, match="trailing dimension"):
            predictor += lsl.Var.new_value(jnp.ones((5, 2)), name="wrong_shape")

        other = gam.MultivariateContribution(
            latent=lsl.Var.new_value(jnp.ones((5, 3))),
            dimension_reparam=lsl.Value(jnp.eye(3)),
            dimension_penalty=lsl.Value(2.0 * jnp.eye(3)),
            name="other",
        )
        with pytest.raises(ValueError, match="different cross-dimensional penalty"):
            predictor += other

    def test_explicit_intercept_is_basis_free_and_consolidates(self) -> None:
        predictor = gam.MVAdditivePredictor.from_random_walk(
            "delta", ndim=4, intercept=False
        ).constrain("sumzero_coef")
        builder = gam.MVTermBuilder.from_predictor(
            predictor,
            gam.TermBuilder.from_dict({"x": jnp.arange(6.0)}),
        )
        intercept = builder.intercept(name="d0", scale=1.0)
        predictor += intercept

        assert intercept.input_obs == {}
        assert intercept.value.shape == (4,)
        assert intercept.latent.value.shape == (3,)
        assert not isinstance(intercept, gam.Basis)

        model = lsl.Model([predictor])
        consolidated, bases = consolidate_bases(model)
        assert predictor.name in consolidated.vars
        assert intercept.name in consolidated.vars
        assert not bases.vars

    def test_explicit_builder_intercept_can_replace_predictor_intercept(self) -> None:
        predictor = gam.MVAdditivePredictor.from_identity(
            "delta", ndim=3, intercept=False
        ).constrain("sumzero_coef")
        builder = gam.MVTermBuilder.from_predictor(
            predictor,
            gam.TermBuilder.from_dict({"x": jnp.arange(6.0)}),
        )
        intercept = builder.intercept(name="d0", scale=1.0)
        predictor.intercept = intercept

        assert predictor.intercept is intercept
        assert predictor.value.shape == (3,)
        model = lsl.Model([predictor])
        assert intercept.name in model.vars
