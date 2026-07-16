import inspect

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pandas as pd
import pytest

import liesel_gam as gam
from liesel_gam.term_builder import _find_parameter


def _data(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": jnp.linspace(0.0, 1.0, n),
            "z": jnp.linspace(1.0, 2.0, n),
            "group": pd.Categorical((["a", "b", "c"] * n)[:n]),
        }
    )


def _basis(x):
    x = jnp.squeeze(x, axis=-1)
    return jnp.column_stack((jnp.ones_like(x), x))


def _scalar_term(builder: gam.TermBuilder, x: str):
    return builder.f(
        x,
        basis_fn=_basis,
        penalty=jnp.eye(2),
        scale=1.0,
        use_callback=False,
    )


def _mv_term(builder: gam.MVTermBuilder, x: str, **kwargs):
    return builder.f(
        x,
        basis_fn=_basis,
        penalty=jnp.eye(2),
        scale=1.0,
        use_callback=False,
        **kwargs,
    )


class TestMVTermBuilder:
    def test_constructors_and_penalty_scaling(self) -> None:
        data = _data()
        raw_penalty = 2.0 * jnp.eye(3)

        scaled = gam.MVTermBuilder.from_df(data, raw_penalty)
        assert scaled.ndim == 3
        assert scaled.latent_ndim == 3
        assert jnp.allclose(scaled.dimension_penalty.value, jnp.eye(3))

        unscaled = gam.MVTermBuilder.from_dict(
            data.to_dict("list"), raw_penalty, scale_penalty=False
        )
        assert jnp.allclose(unscaled.dimension_penalty.value, raw_penalty)

        term_builder = gam.TermBuilder.from_df(data)
        wrapped = gam.MVTermBuilder.from_term_builder(
            term_builder, raw_penalty, scale_penalty=False
        )
        assert wrapped.registry is term_builder.registry
        assert wrapped.names is term_builder.names

    def test_from_predictor_shares_constraint_objects(self) -> None:
        predictor = gam.MVAdditivePredictor.from_random_walk(
            "delta", ndim=4, intercept=False
        ).constrain("sumzero_coef")
        builder = gam.MVTermBuilder.from_predictor(
            predictor, gam.TermBuilder.from_df(_data())
        )
        term = _mv_term(builder, "x", dimension_scale=1.0)

        assert builder.dimension_penalty is predictor.penalty
        assert builder.dimension_reparam is predictor.dimension_reparam
        assert term.dimension_penalty is predictor.penalty
        assert term.dimension_reparam is predictor.dimension_reparam
        assert term.latent.value.shape == (12, 3)
        assert term.value.shape == (12, 4)

        predictor += term
        assert predictor.value.shape == (12, 4)
        model = lsl.Model([predictor])
        assert term.name in model.vars
        assert term.latent.name in model.vars

    def test_cross_dimension_scale_function_and_inference(self) -> None:
        calls = []

        def dimension_scale():
            calls.append(True)
            return gam.ScaleIG(
                value=2.0,
                concentration=1.0,
                scale=0.01,
                name="{x}",
                variance_name="{x}^2",
            )

        builder = gam.MVTermBuilder.from_df(
            _data(),
            jnp.eye(3),
            default_dimension_scale_fn=dimension_scale,
        )
        term = _mv_term(builder, "x")

        assert calls == [True]
        assert term.dimension_scale.value == pytest.approx(2.0)
        assert "\\psi" in term.dimension_scale.name
        scale_parameter = _find_parameter(term.dimension_scale)
        assert isinstance(scale_parameter.inference, gs.MCMCSpec)
        assert scale_parameter.inference.kernel is gs.HMCKernel

    def test_no_penalty_uses_no_cross_scale(self) -> None:
        builder = gam.MVTermBuilder.from_df(_data(), jnp.zeros((3, 3)))
        term = _mv_term(builder, "x")
        second = _mv_term(builder, "z")
        assert term.dimension_scale is None
        assert second.dimension_scale is None
        lsl.Model([term, second])

        with pytest.raises(ValueError, match="not identified"):
            _mv_term(builder, "x", dimension_scale=2.0)

    def test_mirrors_public_term_methods(self) -> None:
        expected = {
            "lin",
            "slin",
            "cr",
            "cs",
            "cc",
            "bs",
            "ps",
            "np",
            "cp",
            "ri",
            "rs",
            "vc",
            "mrf",
            "f",
            "kriging",
            "tp",
            "ts",
            "tx",
            "tf",
        }
        public = {
            name
            for name, method in inspect.getmembers(
                gam.MVTermBuilder, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        }
        assert expected <= public
        assert "intercept" in public

    def test_linear_terms_keep_formula_metadata(self) -> None:
        builder = gam.MVTermBuilder.from_df(_data(), jnp.eye(3))
        lin = builder.lin("x + z", dimension_scale=1.0)
        slin = builder.slin("x + z", scale=1.0, dimension_scale=1.0)

        assert isinstance(lin, gam.MultivariateStrctLinTerm)
        assert isinstance(slin, gam.MultivariateStrctLinTerm)
        assert lin.column_names == ["x", "z"]
        assert slin.column_names == ["x", "z"]
        assert lin.value.shape == (12, 3)
        lsl.Model([lin, slin])

    def test_ps_convenience_method(self) -> None:
        builder = gam.MVTermBuilder.from_df(_data(), jnp.eye(3))
        term = builder.ps(
            "x", k=5, scale=1.0, dimension_scale=1.0
        )

        assert term.latent.value.shape == (12, 3)
        assert term.value.shape == (12, 3)
        lsl.Model([term])

    def test_factor_scale_is_rejected(self) -> None:
        builder = gam.MVTermBuilder.from_df(_data(), jnp.eye(3))
        with pytest.raises(NotImplementedError, match="factor_scale"):
            builder.f(
                "x",
                basis_fn=_basis,
                penalty=jnp.eye(2),
                factor_scale=True,
                use_callback=False,
            )

    def test_tx_accepts_scalar_and_multivariate_marginals(self) -> None:
        data = _data()
        scalar_builder = gam.TermBuilder.from_df(data)
        builder = gam.MVTermBuilder.from_term_builder(
            scalar_builder, jnp.eye(3)
        )
        scalar = _scalar_term(scalar_builder, "x")
        multivariate = _mv_term(builder, "z", dimension_scale=1.0)

        term = builder.tx(scalar, multivariate, dimension_scale=1.0)
        assert len(term.marginal_terms) == 2
        assert term.latent.value.shape == (12, 3)
        assert term.value.shape == (12, 3)
        lsl.Model([term])

    def test_tf_accepts_mixed_marginals_and_builds_unique_graph(self) -> None:
        data = _data()
        scalar_builder = gam.TermBuilder.from_df(data)
        builder = gam.MVTermBuilder.from_term_builder(
            scalar_builder, jnp.eye(3)
        )

        scalar = _scalar_term(scalar_builder, "x")
        multivariate = _mv_term(builder, "z", dimension_scale=1.0)
        tf1 = builder.tf(
            scalar,
            multivariate,
            dimension_scale=1.0,
        )
        tf2 = builder.tf(
            scalar,
            multivariate,
            dimension_scale=1.0,
        )

        assert sorted(tf1.terms_by_order) == [1, 2]
        assert len(tf1._terms_list) == 3
        assert tf1.latent.value.shape == (12, 3)
        assert tf1.value.shape == (12, 3)
        lsl.Model([tf1, tf2])

    def test_tf_groups_terms_by_order(self) -> None:
        data = _data()
        scalar_builder = gam.TermBuilder.from_df(data)
        builder = gam.MVTermBuilder.from_term_builder(
            scalar_builder, jnp.eye(3)
        )
        term = builder.tf(
            (sx := _scalar_term(scalar_builder, "x")),
            (sz := _scalar_term(scalar_builder, "z")),
            dimension_scale=1.0,
            group_terms_by_order=True,
        )
        second = builder.tf(
            sx,
            sz,
            dimension_scale=1.0,
            group_terms_by_order=True,
        )

        model = lsl.Model([term, second])
        assert set(term.term_groups) == {1, 2}
        assert all(group.name in model.vars for group in term.term_groups.values())

    def test_tensor_rejects_incompatible_multivariate_marginal(self) -> None:
        data = _data()
        scalar_builder = gam.TermBuilder.from_df(data)
        identity = gam.MVTermBuilder.from_term_builder(
            scalar_builder, jnp.eye(3)
        )
        random_walk = gam.MVTermBuilder.from_term_builder(
            scalar_builder,
            jnp.diff(jnp.eye(3), axis=0).T @ jnp.diff(jnp.eye(3), axis=0),
        )

        incompatible = _mv_term(random_walk, "z", dimension_scale=1.0)
        with pytest.raises(ValueError, match="different dimension penalty"):
            identity.tf(_scalar_term(scalar_builder, "x"), incompatible)

    def test_tensor_input_validation(self) -> None:
        data = _data()
        scalar_builder = gam.TermBuilder.from_df(data)
        builder = gam.MVTermBuilder.from_term_builder(
            scalar_builder, jnp.eye(3)
        )
        scalar = _scalar_term(scalar_builder, "x")

        with pytest.raises(ValueError, match="At least one tensor marginal"):
            builder.tf()
        with pytest.raises(ValueError, match="order must contain"):
            builder.tf(scalar, order=())
        with pytest.raises(ValueError, match="unique integers"):
            builder.tf(scalar, order=(2,))

    def test_reparameterization_must_have_full_column_rank(self) -> None:
        scalar_builder = gam.TermBuilder.from_df(_data())
        scalar = _scalar_term(scalar_builder, "x")

        with pytest.raises(ValueError, match="full column rank"):
            gam.MultivariateStrctTerm(
                scalar,
                dimension_penalties=[jnp.eye(2)],
                dimension_scales=[1.0],
                dimension_reparam=jnp.zeros((3, 2)),
            )

    def test_low_level_tensor_applies_common_scale_before_subterms(self) -> None:
        scalar_builder = gam.TermBuilder.from_df(_data())
        sx = _scalar_term(scalar_builder, "x")
        sz = _scalar_term(scalar_builder, "z")
        common_scale = lsl.Var.new_value(2.0, name="common_scale")
        dimension_scale = lsl.Var.new_value(1.0, name="dimension_scale")

        term = gam.MultivariateTPTerm(
            sx,
            sz,
            common_scale=common_scale,
            dimension_penalties=[jnp.eye(3)],
            dimension_scales=[dimension_scale],
        )

        for subterm in term._terms_list:
            assert all(scale is common_scale for scale in subterm.marginal_scales)

    def test_random_slope_and_varying_coefficient(self) -> None:
        data = _data()
        scalar_builder = gam.TermBuilder.from_df(data)
        builder = gam.MVTermBuilder.from_term_builder(
            scalar_builder, jnp.eye(3)
        )

        random_slope = builder.rs(
            "x", "group", scale=1.0, dimension_scale=1.0
        )
        varying = builder.vc(
            "x", _scalar_term(scalar_builder, "z"), dimension_scale=1.0
        )

        assert random_slope.value.shape == (12, 3)
        assert varying.value.shape == (12, 3)
        lsl.Model([random_slope, varying])
