import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_gam.basis import Basis
from liesel_gam.iwls_proposals import (
    IWLSWeights,
    TwoPieceStudentTDfIWLSProposal,
    TwoPieceStudentTIWLSWeights,
    TwoPieceStudentTLocIWLSProposal,
    TwoPieceStudentTScaleIWLSProposal,
    TwoPieceStudentTSkewnessIWLSProposal,
    apply_two_piece_student_t_iwls_spec_df,
    apply_two_piece_student_t_iwls_spec_loc,
    apply_two_piece_student_t_iwls_spec_scale,
    apply_two_piece_student_t_iwls_spec_skewness,
    two_piece_student_t_iwls_spec_df,
    two_piece_student_t_iwls_spec_loc,
    two_piece_student_t_iwls_spec_scale,
    two_piece_student_t_iwls_spec_skewness,
)
from liesel_gam.predictor import AdditivePredictor
from liesel_gam.term import StrctTerm


class DictModel:
    def extract_position(self, position_keys, model_state):
        return {key: model_state[key] for key in position_keys}


def _term_and_parameters():
    x = jnp.linspace(-1.0, 1.0, 8)
    basis = Basis(
        jnp.column_stack([jnp.ones_like(x), x]),
        xname="x_skew_t",
        penalty=jnp.eye(2),
        use_callback=False,
    )
    term = StrctTerm.f(
        basis,
        scale=lsl.Var.new_param(jnp.array(1.5), name="tau_skew_t"),
    )
    df = lsl.Var.new_param(jnp.array(5.0), name="nu")
    scale = lsl.Var.new_param(jnp.array(1.7), name="sigma")
    skewness = lsl.Var.new_param(jnp.array(2.2), name="gamma")
    return term, df, scale, skewness


def test_two_piece_student_t_weights_match_tfp_mean_hessian():
    df = jnp.array(5.0, dtype=jnp.float32)
    scale = jnp.array(1.7, dtype=jnp.float32)
    skewness = jnp.array(2.2, dtype=jnp.float32)
    t_key, side_key = jax.random.split(jax.random.key(42))
    standard_t = jax.random.t(t_key, df, shape=(50_000,), dtype=df.dtype)
    left = jax.random.bernoulli(
        side_key,
        1.0 / (1.0 + jnp.square(skewness)),
        shape=standard_t.shape,
    )
    two_piece_t = jnp.where(
        left,
        -jnp.abs(standard_t) / skewness,
        jnp.abs(standard_t) * skewness,
    )
    samples = scale * two_piece_t

    def log_prob(theta, y):
        return tfd.TwoPieceStudentT(
            df=jnp.exp(theta[2]),
            loc=theta[0],
            scale=jnp.exp(theta[1]),
            skewness=jnp.exp(theta[3]),
        ).log_prob(y)

    theta = jnp.array([0.0, jnp.log(scale), jnp.log(df), jnp.log(skewness)])
    mean_hessian_diagonal = jax.jit(
        jax.vmap(lambda y: -jnp.diag(jax.hessian(log_prob)(theta, y)))
    )(samples).mean(axis=0)

    state = {"df": df, "scale": scale, "skewness": skewness}
    model = DictModel()
    expected = jnp.stack(
        [
            IWLSWeights.two_piece_student_t_loc()(model, state),
            IWLSWeights.two_piece_student_t_scale()(model, state),
            IWLSWeights.two_piece_student_t_df()(model, state),
            IWLSWeights.two_piece_student_t_skewness()(model, state),
        ]
    )

    assert jnp.allclose(mean_hessian_diagonal, expected, rtol=0.03, atol=0.003)


def test_two_piece_student_t_weight_aliases_and_large_df_are_positive():
    state = {
        "df": jnp.array(10_000.0, dtype=jnp.float32),
        "scale": jnp.array(2.0, dtype=jnp.float32),
        "skewness": jnp.array(0.5, dtype=jnp.float32),
    }
    model = DictModel()

    for name in ("loc", "scale", "df", "skewness"):
        generic = getattr(IWLSWeights, f"two_piece_student_t_{name}")()(model, state)
        alias = getattr(TwoPieceStudentTIWLSWeights, name)()(model, state)
        assert jnp.isfinite(generic)
        assert generic > 0.0
        assert alias == pytest.approx(generic)


@pytest.mark.parametrize(
    ("spec_fn", "proposal_type", "spec_kwargs"),
    [
        (
            two_piece_student_t_iwls_spec_loc,
            TwoPieceStudentTLocIWLSProposal,
            {"df_name": "nu", "scale_name": "sigma"},
        ),
        (
            two_piece_student_t_iwls_spec_scale,
            TwoPieceStudentTScaleIWLSProposal,
            {"df_name": "nu"},
        ),
        (
            two_piece_student_t_iwls_spec_df,
            TwoPieceStudentTDfIWLSProposal,
            {"df_name": "nu"},
        ),
        (
            two_piece_student_t_iwls_spec_skewness,
            TwoPieceStudentTSkewnessIWLSProposal,
            {"df_name": "nu", "skewness_name": "gamma"},
        ),
    ],
    ids=("loc", "scale", "df", "skewness"),
)
def test_two_piece_student_t_specs_build_parameter_specific_proposals(
    spec_fn,
    proposal_type,
    spec_kwargs,
):
    term, df, scale, skewness = _term_and_parameters()
    model = lsl.Model([term, df, scale, skewness])

    spec = spec_fn(term, fallback_chol_info=None, **spec_kwargs)
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)
    proposal = kernel.chol_info_fn.__self__

    assert isinstance(spec, gs.MCMCSpec)
    assert isinstance(proposal, proposal_type)
    assert proposal.working_weights(model.state) > 0.0
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)


@pytest.mark.parametrize(
    ("apply_fn", "proposal_type", "apply_kwargs"),
    [
        (
            apply_two_piece_student_t_iwls_spec_loc,
            TwoPieceStudentTLocIWLSProposal,
            {"df_name": "nu", "scale_name": "sigma"},
        ),
        (
            apply_two_piece_student_t_iwls_spec_scale,
            TwoPieceStudentTScaleIWLSProposal,
            {"df_name": "nu"},
        ),
        (
            apply_two_piece_student_t_iwls_spec_df,
            TwoPieceStudentTDfIWLSProposal,
            {"df_name": "nu"},
        ),
        (
            apply_two_piece_student_t_iwls_spec_skewness,
            TwoPieceStudentTSkewnessIWLSProposal,
            {"df_name": "nu", "skewness_name": "gamma"},
        ),
    ],
    ids=("loc", "scale", "df", "skewness"),
)
def test_apply_two_piece_student_t_specs_assigns_parameter_specific_proposals(
    apply_fn,
    proposal_type,
    apply_kwargs,
):
    term, df, scale, skewness = _term_and_parameters()
    predictor = AdditivePredictor("eta", intercept=False)
    predictor += term

    apply_fn(predictor, fallback_chol_info=None, **apply_kwargs)
    model = lsl.Model([predictor, df, scale, skewness])
    spec = term.coef.inference
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert isinstance(kernel.chol_info_fn.__self__, proposal_type)
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)
