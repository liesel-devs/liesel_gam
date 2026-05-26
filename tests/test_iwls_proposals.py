import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pytest

from liesel_gam.basis import Basis
from liesel_gam.iwls_proposals import (
    GaussianIWLS,
    GaussianLocIWLSProposal,
    GaussianScaleIWLSProposal,
    IWLSProposal,
    apply_gaussian_iwls_spec_loc,
    apply_gaussian_iwls_spec_scale,
    gaussian_iwls_spec_loc,
    gaussian_iwls_spec_scale,
)
from liesel_gam.predictor import AdditivePredictor
from liesel_gam.term import MRFTerm, RITerm, StrctLinTerm, StrctTerm


class DictModel:
    def extract_position(self, position_keys, model_state):
        return {key: model_state[key] for key in position_keys}


def _state(obs_scale):
    Z = jnp.array(
        [
            [1.0, -0.5],
            [0.25, 0.75],
            [1.5, 0.25],
        ],
        dtype=jnp.float32,
    )
    penalty = jnp.array([[2.0, 0.1], [0.1, 1.0]], dtype=Z.dtype)
    state = {
        "B": Z,
        "tau": jnp.array(1.25, dtype=Z.dtype),
        "obs_scale": jnp.asarray(obs_scale, dtype=Z.dtype),
    }
    return Z, penalty, state


def _expected_precision(Z, penalty, weights, smooth_scale):
    w = jnp.asarray(weights, dtype=Z.dtype)
    ZW = Z * (w[:, None] if w.ndim == 1 else w)
    ZTWZ = Z.T @ ZW

    eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)
    inv_scale2 = 1.0 / jnp.clip(smooth_scale, min=eps) ** 2

    P = ZTWZ + inv_scale2 * penalty
    jitter = 1e-6 * jnp.mean(jnp.diag(P)) * jnp.eye(*P.shape, dtype=P.dtype)
    return P + jitter


def _loc_info(penalty, scale_factored=False):
    return GaussianLocIWLSProposal(
        basis_name="B",
        smooth_name="f(x)",
        smooth_scale_name="tau",
        scale_name="obs_scale",
        penalty=penalty,
        model=DictModel(),
        n=3,
        scale_factored=scale_factored,
    )


def _scale_info(penalty, scale_factored=False):
    return GaussianScaleIWLSProposal(
        basis_name="B",
        smooth_name="f(x)",
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        n=3,
        scale_factored=scale_factored,
    )


def _term_and_scale(factor_scale=False):
    x = jnp.linspace(-1.0, 1.0, 6)
    basis_matrix = jnp.column_stack([jnp.ones_like(x), x])
    penalty = jnp.array([[1.0, 0.2], [0.2, 2.0]], dtype=basis_matrix.dtype)
    basis = Basis(
        basis_matrix,
        xname="x",
        penalty=penalty,
        use_callback=False,
    )
    smooth_scale = lsl.Var.new_param(jnp.array(1.5), name="smooth_scale")
    term = StrctTerm.f(basis, scale=smooth_scale, factor_scale=factor_scale)
    obs_scale = lsl.Var.new_param(jnp.array(2.0), name="obs_scale")
    return term, obs_scale


def _term_model(factor_scale=False):
    term, obs_scale = _term_and_scale(factor_scale=factor_scale)
    model = lsl.Model([term, obs_scale])
    return term, model


def _model_for_term(term):
    obs_scale = lsl.Var.new_param(jnp.array(2.0), name="obs_scale")
    model = lsl.Model([term, obs_scale])
    return model


def _structured_term_class_examples():
    x = jnp.linspace(-1.0, 1.0, 6)
    basis_matrix = jnp.column_stack([jnp.ones_like(x), x])
    penalty = jnp.array([[1.0, 0.2], [0.2, 2.0]], dtype=basis_matrix.dtype)

    basis = Basis(basis_matrix, xname="x_strct", penalty=penalty, use_callback=False)
    yield StrctTerm.f(basis, scale=lsl.Var.new_param(jnp.array(1.5), name="tau_strct"))

    basis = Basis(basis_matrix, xname="x_mrf", penalty=penalty, use_callback=False)
    yield MRFTerm.f(basis, scale=lsl.Var.new_param(jnp.array(1.5), name="tau_mrf"))

    basis = Basis(basis_matrix, xname="x_lin", penalty=penalty, use_callback=False)
    yield StrctLinTerm(
        basis,
        penalty=basis.penalty,
        scale=lsl.Var.new_param(jnp.array(1.5), name="tau_lin"),
        name="strct_lin",
    )

    group = jnp.array([0, 1, 0, 1])
    ri_basis = Basis(
        group,
        xname="group",
        penalty=jnp.eye(2),
        use_callback=False,
    )
    yield RITerm(
        ri_basis,
        penalty=ri_basis.penalty,
        scale=lsl.Var.new_param(jnp.array(1.5), name="tau_ri"),
        name="ri",
    )


def test_loc_working_weights_clip_observation_scale_at_machine_epsilon():
    _, penalty, _ = _state(jnp.array(1.0))
    state = {"obs_scale": jnp.array([0.0, 2.0], dtype=jnp.float32)}
    info = _loc_info(penalty)

    eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
    expected = 1.0 / jnp.clip(state["obs_scale"], min=eps) ** 2

    actual = info.working_weights(state)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.allclose(actual, expected)


def test_gaussian_iwls_loc_factory_clips_observation_scale_at_machine_epsilon():
    state = {"obs_scale": jnp.array([0.0, 2.0], dtype=jnp.float32)}

    eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
    expected = 1.0 / jnp.clip(state["obs_scale"], min=eps) ** 2

    actual = GaussianIWLS.loc(scale_name="obs_scale")(DictModel(), state)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize(
    "obs_scale",
    (
        pytest.param(jnp.array(2.0, dtype=jnp.float32), id="scalar"),
        pytest.param(jnp.array([1.0, 2.0, 4.0], dtype=jnp.float32), id="vector"),
    ),
)
def test_loc_precision_matches_weighted_crossproduct(obs_scale):
    Z, penalty, state = _state(obs_scale)
    info = _loc_info(penalty)

    eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)
    weights = 1.0 / jnp.clip(obs_scale, min=eps) ** 2
    expected = _expected_precision(Z, penalty, weights, state["tau"])

    assert jnp.allclose(info.precision(state), expected, rtol=1e-6, atol=1e-6)


def test_loc_precision_is_jittable():
    Z, penalty, state = _state(jnp.array([1.0, 2.0, 4.0], dtype=jnp.float32))
    info = _loc_info(penalty)

    expected = info.precision(state)
    actual = jax.jit(lambda model_state: info.precision(model_state))(state)

    assert actual.shape == (Z.shape[1], Z.shape[1])
    assert jnp.allclose(actual, expected)


def test_loc_chol_info_matches_precision_cholesky():
    _, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    info = _loc_info(penalty)

    chol = info.chol_info(state)
    expected = jnp.linalg.cholesky(info.precision(state))

    assert jnp.allclose(chol, expected)
    assert jnp.allclose(chol, jnp.tril(chol))


def test_loc_chol_info_rejects_scale_factored():
    _, penalty, _ = _state(jnp.array(2.0, dtype=jnp.float32))

    with pytest.raises(ValueError, match="scale-factored"):
        _loc_info(penalty, scale_factored=True)


def test_scale_working_weights_are_constant_two():
    _, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    info = _scale_info(penalty)

    actual = info.working_weights(state)

    assert actual.shape == ()
    assert actual == pytest.approx(2.0)


def test_gaussian_iwls_scale_factory_returns_constant_two():
    _, _, state = _state(jnp.array(2.0, dtype=jnp.float32))

    actual = GaussianIWLS.scale()(DictModel(), state)

    assert actual.shape == ()
    assert actual == pytest.approx(2.0)


def test_iwls_proposal_uses_supplied_working_weights_function():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    weights = jnp.array([0.5, 1.5, 2.5], dtype=Z.dtype)

    def working_weights(model, model_state):
        return model.extract_position(["custom_weights"], model_state)["custom_weights"]

    state = state | {"custom_weights": weights}
    info = IWLSProposal(
        basis_name="B",
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=working_weights,
    )

    expected = _expected_precision(Z, penalty, weights, state["tau"])

    assert jnp.allclose(info.working_weights(state), weights)
    assert jnp.allclose(info.precision(state), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("term", list(_structured_term_class_examples()))
def test_iwls_proposal_from_term_extracts_geometry_from_supported_terms(term):
    model = _model_for_term(term)

    proposal = IWLSProposal.from_term(term, GaussianIWLS.scale())

    assert proposal.basis_name == term.basis.name
    assert proposal.smooth_scale_name == term.scale.name
    assert proposal.penalty is term.basis.penalty.value
    assert proposal.model is model
    assert proposal.scale_factored is False


def test_iwls_proposal_from_term_rejects_terms_without_model():
    term, _ = _term_and_scale()

    with pytest.raises(ValueError, match="attached to a model"):
        IWLSProposal.from_term(term, GaussianIWLS.scale())


def test_gaussian_iwls_proposals_can_be_constructed_from_term():
    term, model = _term_model()

    loc_proposal = GaussianLocIWLSProposal.from_term(term, scale_name="obs_scale")
    scale_proposal = GaussianScaleIWLSProposal.from_term(term)

    assert loc_proposal.basis_name == term.basis.name
    assert loc_proposal.smooth_name == term.name
    assert loc_proposal.smooth_scale_name == term.scale.name
    assert loc_proposal.scale_name == "obs_scale"
    assert loc_proposal.model is model
    assert loc_proposal.n == term.value.shape[0]

    assert scale_proposal.basis_name == term.basis.name
    assert scale_proposal.smooth_name == term.name
    assert scale_proposal.smooth_scale_name == term.scale.name
    assert scale_proposal.model is model
    assert scale_proposal.n == term.value.shape[0]


def test_iwls_proposal_kernel_factory_uses_bound_proposal():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    weights = jnp.array([0.5, 1.5, 2.5], dtype=Z.dtype)

    def working_weights(model, model_state):
        return model.extract_position(["custom_weights"], model_state)["custom_weights"]

    state = state | {"custom_weights": weights}
    proposal = IWLSProposal(
        basis_name="B",
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=working_weights,
    )
    kernel_factory = proposal.kernel_factory()
    kernel = kernel_factory(["coef"], fallback_chol_info=None)

    assert isinstance(kernel, gs.IWLSKernel)
    assert kernel.position_keys == ("coef",)
    assert kernel.initial_step_size == pytest.approx(1.0)
    assert kernel.da_tune_step_size is False
    assert kernel.fallback_chol_info is None
    assert kernel.chol_info_fn.__self__ is proposal
    assert jnp.allclose(
        kernel.chol_info_fn(state),
        jnp.linalg.cholesky(_expected_precision(Z, penalty, weights, state["tau"])),
    )


def test_scale_precision_matches_weighted_crossproduct():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    info = _scale_info(penalty)

    expected = _expected_precision(
        Z, penalty, jnp.array(2.0, dtype=Z.dtype), state["tau"]
    )

    assert jnp.allclose(info.precision(state), expected, rtol=1e-6, atol=1e-6)


def test_scale_chol_info_matches_precision_cholesky():
    _, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    info = _scale_info(penalty)

    chol = info.chol_info(state)
    expected = jnp.linalg.cholesky(info.precision(state))

    assert jnp.allclose(chol, expected)
    assert jnp.allclose(chol, jnp.tril(chol))


def test_scale_chol_info_rejects_scale_factored():
    _, penalty, _ = _state(jnp.array(2.0, dtype=jnp.float32))

    with pytest.raises(ValueError, match="scale-factored"):
        _scale_info(penalty, scale_factored=True)


def test_gaussian_iwls_spec_loc_builds_untuned_kernel_with_custom_chol_info():
    term, model = _term_model()

    spec = gaussian_iwls_spec_loc(
        term,
        scale_name="obs_scale",
        fallback_chol_info=None,
    )
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)
    proposal = kernel.chol_info_fn.__self__

    assert isinstance(spec, gs.MCMCSpec)
    assert spec.kernel_kwargs == {"term": term}
    assert isinstance(kernel, gs.IWLSKernel)
    assert kernel.position_keys == (term.coef.name,)
    assert kernel.initial_step_size == pytest.approx(1.0)
    assert kernel.da_tune_step_size is False
    assert kernel.fallback_chol_info is None
    assert isinstance(proposal, GaussianLocIWLSProposal)
    assert proposal.basis_name == term.basis.name
    assert proposal.smooth_name == term.name
    assert proposal.smooth_scale_name == term.scale.name
    assert proposal.scale_name == "obs_scale"
    assert proposal.model is model
    assert proposal.n == term.value.shape[0]
    assert proposal.scale_factored is False
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)


def test_gaussian_iwls_spec_loc_forwards_kernel_kwargs():
    term, _ = _term_model()

    spec = gaussian_iwls_spec_loc(
        term,
        scale_name="obs_scale",
        initial_step_size=0.25,
        da_tune_step_size=True,
        fallback_chol_info="chol_of_modified_info",
    )
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert kernel.initial_step_size == pytest.approx(0.25)
    assert kernel.da_tune_step_size is True
    assert kernel.fallback_chol_info == "chol_of_modified_info"


def test_gaussian_iwls_spec_loc_rejects_factored_terms():
    term, _ = _term_model(factor_scale=True)

    with pytest.raises(ValueError, match="scale-factored"):
        gaussian_iwls_spec_loc(term, scale_name="obs_scale")


def test_gaussian_iwls_spec_scale_builds_untuned_kernel_with_custom_chol_info():
    term, model = _term_model()

    spec = gaussian_iwls_spec_scale(term, fallback_chol_info=None)
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)
    proposal = kernel.chol_info_fn.__self__

    assert isinstance(spec, gs.MCMCSpec)
    assert spec.kernel_kwargs == {"term": term}
    assert isinstance(kernel, gs.IWLSKernel)
    assert kernel.position_keys == (term.coef.name,)
    assert kernel.initial_step_size == pytest.approx(1.0)
    assert kernel.da_tune_step_size is False
    assert kernel.fallback_chol_info is None
    assert isinstance(proposal, GaussianScaleIWLSProposal)
    assert proposal.basis_name == term.basis.name
    assert proposal.smooth_name == term.name
    assert proposal.smooth_scale_name == term.scale.name
    assert proposal.model is model
    assert proposal.n == term.value.shape[0]
    assert proposal.scale_factored is False
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)


def test_gaussian_iwls_spec_scale_rejects_factored_terms():
    term, _ = _term_model(factor_scale=True)

    with pytest.raises(ValueError, match="scale-factored"):
        gaussian_iwls_spec_scale(term)


def test_apply_gaussian_iwls_spec_loc_assigns_structured_terms_only():
    term, obs_scale = _term_and_scale()
    offset = lsl.Var.new_value(jnp.ones_like(term.value), name="offset")
    predictor = AdditivePredictor("loc", intercept=False)
    predictor += term, offset

    apply_gaussian_iwls_spec_loc(
        predictor,
        scale_name="obs_scale",
        fallback_chol_info=None,
    )
    model = lsl.Model([predictor, obs_scale])
    spec = term.coef.inference
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert isinstance(spec, gs.MCMCSpec)
    assert getattr(offset, "inference", None) is None
    assert isinstance(kernel.chol_info_fn.__self__, GaussianLocIWLSProposal)
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)


def test_apply_gaussian_iwls_spec_loc_rejects_factored_terms():
    term, _ = _term_and_scale(factor_scale=True)
    predictor = AdditivePredictor("loc", intercept=False)
    predictor += term

    with pytest.raises(ValueError, match="scale-factored"):
        apply_gaussian_iwls_spec_loc(predictor, scale_name="obs_scale")


def test_apply_gaussian_iwls_spec_scale_assigns_structured_terms_only():
    term, obs_scale = _term_and_scale()
    offset = lsl.Var.new_value(jnp.ones_like(term.value), name="offset")
    predictor = AdditivePredictor("scale", intercept=False)
    predictor += term, offset

    apply_gaussian_iwls_spec_scale(predictor, fallback_chol_info=None)
    model = lsl.Model([predictor, obs_scale])
    spec = term.coef.inference
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert isinstance(spec, gs.MCMCSpec)
    assert getattr(offset, "inference", None) is None
    assert isinstance(kernel.chol_info_fn.__self__, GaussianScaleIWLSProposal)
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)


def test_apply_gaussian_iwls_spec_scale_rejects_factored_terms():
    term, _ = _term_and_scale(factor_scale=True)
    predictor = AdditivePredictor("scale", intercept=False)
    predictor += term

    with pytest.raises(ValueError, match="scale-factored"):
        apply_gaussian_iwls_spec_scale(predictor)
