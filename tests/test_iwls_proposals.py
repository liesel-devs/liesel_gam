import logging

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_gam.basis import Basis
from liesel_gam.category_mapping import CategoryMapping
from liesel_gam.iwls_proposals import (
    GaussianIWLSWeights,
    GaussianLocIWLSProposal,
    GaussianScaleIWLSProposal,
    IWLSProposal,
    IWLSWeights,
    apply_gaussian_iwls_spec_loc,
    apply_gaussian_iwls_spec_scale,
    gaussian_iwls_spec_loc,
    gaussian_iwls_spec_scale,
)
from liesel_gam.predictor import AdditivePredictor
from liesel_gam.term import (
    IndexingTerm,
    MRFTerm,
    RITerm,
    StrctInteractionTerm,
    StrctLinTerm,
    StrctTerm,
)


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


def _gaussian_eta_liesel_state(
    eta=None,
    y=None,
    scale=2.0,
    *,
    basis=None,
    smooth_scale=None,
):
    eta = jnp.asarray(
        eta if eta is not None else jnp.array([-1.0, 0.25, 1.5]),
        dtype=jnp.float32,
    )
    y = jnp.asarray(
        y if y is not None else jnp.array([0.5, -0.75, 2.0]),
        dtype=eta.dtype,
    )
    scale = jnp.asarray(scale, dtype=eta.dtype)

    eta_var = lsl.Var.new_param(eta, name="eta")
    scale_var = lsl.Var.new_param(scale, name="scale")
    y_var = lsl.Var.new_obs(
        y,
        lsl.Dist(tfd.Normal, loc=eta_var, scale=scale_var),
        name="y",
    )

    nodes = [y_var, eta_var, scale_var]
    values = {"eta": eta, "y": y, "scale": scale}

    if basis is not None:
        basis = jnp.asarray(basis, dtype=eta.dtype)
        nodes.append(lsl.Var.new_obs(basis, name="B"))
        values["B"] = basis

    if smooth_scale is not None:
        smooth_scale = jnp.asarray(smooth_scale, dtype=eta.dtype)
        nodes.append(lsl.Var.new_param(smooth_scale, name="tau"))
        values["tau"] = smooth_scale

    model = lsl.Model(nodes)
    return gs.LieselInterface(model), model.state, values


def _squared_eta_liesel_state(eta=None, y=None, scale=1.0):
    eta = jnp.asarray(
        eta if eta is not None else jnp.array([-2.0, -0.5, 1.5]),
        dtype=jnp.float32,
    )
    y = jnp.asarray(y if y is not None else jnp.zeros_like(eta), dtype=eta.dtype)
    scale = jnp.asarray(scale, dtype=eta.dtype)

    eta_var = lsl.Var.new_param(eta, name="eta")
    scale_var = lsl.Var.new_param(scale, name="scale")
    eta_squared = lsl.Calc(lambda eta: eta**2, eta_var, _name="eta_squared")
    y_var = lsl.Var.new_obs(
        y,
        lsl.Dist(tfd.Normal, loc=eta_squared, scale=scale_var),
        name="y",
    )

    model = lsl.Model([y_var, eta_var, scale_var])
    values = {"eta": eta, "y": y, "scale": scale}
    return gs.LieselInterface(model), model.state, values


def _expected_precision(Z, penalty, weights, smooth_scale):
    w = jnp.asarray(weights, dtype=Z.dtype)
    ZW = Z * (w[:, None] if w.ndim == 1 else w)
    ZTWZ = Z.T @ ZW

    eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)
    inv_scale2 = 1.0 / jnp.clip(smooth_scale, min=eps) ** 2

    P = ZTWZ + inv_scale2 * penalty
    jitter = 1e-6 * jnp.mean(jnp.diag(P)) * jnp.eye(*P.shape, dtype=P.dtype)
    return P + jitter


def _iwls_kernel_default_initial_step_size():
    return gs.IWLSKernel(["coef"]).initial_step_size


def _loc_info(penalty, scale_factored=False):
    Z, _, _ = _state(jnp.array(2.0, dtype=jnp.float32))
    return GaussianLocIWLSProposal(
        basis=Z,
        smooth_name="f(x)",
        smooth_scale_name="tau",
        scale_name="obs_scale",
        penalty=penalty,
        model=DictModel(),
        n=3,
        basis_name="B",
        scale_factored=scale_factored,
    )


def _scale_info(penalty, scale_factored=False):
    Z, _, _ = _state(jnp.array(2.0, dtype=jnp.float32))
    return GaussianScaleIWLSProposal(
        basis=Z,
        smooth_name="f(x)",
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        n=3,
        basis_name="B",
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


def _indexing_term_and_model():
    group = jnp.array([0, 2, 1, 2, 0])
    basis = Basis(group, xname="index_group", penalty=None, use_callback=False)
    penalty = jnp.array(
        [
            [1.5, 0.2, 0.0],
            [0.2, 2.0, 0.1],
            [0.0, 0.1, 1.2],
        ],
        dtype=jnp.float32,
    )
    scale = lsl.Var.new_param(jnp.array(1.25, dtype=penalty.dtype), name="tau_index")
    term = IndexingTerm(
        basis,
        penalty=penalty,
        scale=scale,
        name="index_term",
    )
    obs_scale = lsl.Var.new_param(
        jnp.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=penalty.dtype),
        name="obs_scale",
    )
    model = lsl.Model([term, obs_scale])
    return term, model, group, penalty


def _interaction_term():
    x = jnp.linspace(-1.0, 1.0, 5)
    y = jnp.linspace(0.0, 2.0, 5)
    basis_x = Basis(
        jnp.column_stack([jnp.ones_like(x), x]),
        xname="x_interaction",
        penalty=jnp.eye(2),
        use_callback=False,
    )
    basis_y = Basis(
        jnp.column_stack([jnp.ones_like(y), y]),
        xname="y_interaction",
        penalty=jnp.eye(2),
        use_callback=False,
    )
    term_x = StrctTerm.f(
        basis_x,
        scale=lsl.Var.new_param(jnp.array(1.0), name="tau_interaction_x"),
    )
    term_y = StrctTerm.f(
        basis_y,
        scale=lsl.Var.new_param(jnp.array(1.5), name="tau_interaction_y"),
    )
    return StrctInteractionTerm(term_x, term_y, name="interaction")


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


def test_gaussian_iwls_weights_loc_clips_observation_scale_at_machine_epsilon():
    state = {"obs_scale": jnp.array([0.0, 2.0], dtype=jnp.float32)}

    eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
    expected = 1.0 / jnp.clip(state["obs_scale"], min=eps) ** 2

    actual = IWLSWeights.gaussian_loc(scale_name="obs_scale")(DictModel(), state)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.allclose(actual, expected)


def test_iwls_weights_constant_defaults_to_one():
    state = {"obs_scale": jnp.array([1.0, 2.0], dtype=jnp.float32)}

    actual = IWLSWeights.constant()(DictModel(), state)

    assert actual.shape == ()
    assert actual == pytest.approx(1.0)


def test_iwls_weights_constant_accepts_observation_wise_weights():
    weights = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32)

    actual = IWLSWeights.constant(weights)(DictModel(), {})

    assert jnp.allclose(actual, weights)


def test_iwls_proposal_precision_uses_constant_unit_weights():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    info = IWLSProposal(
        basis=Z,
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=IWLSWeights.constant(),
        basis_name="B",
    )

    expected = _expected_precision(
        Z,
        penalty,
        jnp.array(1.0, dtype=Z.dtype),
        state["tau"],
    )

    assert jnp.allclose(info.precision(state), expected, rtol=1e-6, atol=1e-6)


def test_iwls_proposal_precision_uses_static_basis_matrix():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    changed_state = state | {"B": 10.0 * Z}
    info = IWLSProposal(
        basis=Z,
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=IWLSWeights.constant(),
        basis_name="B",
    )

    expected = _expected_precision(Z, penalty, 1.0, state["tau"])

    assert jnp.allclose(info.precision(changed_state), expected, rtol=1e-6, atol=1e-6)


def test_iwls_weights_score_squared_matches_gaussian_location_score_squared():
    model, state, values = _gaussian_eta_liesel_state()
    expected_score = (values["y"] - values["eta"]) / values["scale"] ** 2
    expected = jnp.clip(jnp.square(expected_score), min=1e-6)

    actual = IWLSWeights.score_squared("eta")(model, state)

    assert actual.shape == values["eta"].shape
    assert jnp.allclose(actual, expected)


def test_iwls_weights_score_squared_matches_nonlinear_log_lik_score_squared():
    model, state, values = _squared_eta_liesel_state()
    expected_score = (
        2.0 * values["eta"] * (values["y"] - values["eta"] ** 2) / values["scale"] ** 2
    )
    expected = jnp.square(expected_score)

    actual = IWLSWeights.score_squared("eta", min_weight=0.0)(model, state)

    assert jnp.allclose(actual, expected)


def test_iwls_weights_score_squared_clips_zero_scores_to_min_weight():
    model, state, values = _gaussian_eta_liesel_state(
        eta=jnp.array([1.0, 2.0, 3.0]),
        y=jnp.array([1.0, 2.0, 3.0]),
    )

    actual = IWLSWeights.score_squared("eta", min_weight=0.25)(model, state)

    assert jnp.allclose(actual, jnp.full_like(values["eta"], 0.25))


def test_iwls_weights_score_squared_clips_large_scores_to_max_weight():
    model, state, values = _gaussian_eta_liesel_state(
        eta=jnp.array([0.0, 1.0, 2.0]),
        y=jnp.array([10.0, -10.0, 20.0]),
        scale=1.0,
    )

    actual = IWLSWeights.score_squared("eta", max_weight=2.5)(model, state)

    assert jnp.allclose(actual, jnp.full_like(values["eta"], 2.5))


def test_iwls_weights_score_squared_preserves_scalar_eta_shape():
    model, state, values = _gaussian_eta_liesel_state(
        eta=jnp.array(1.0),
        y=jnp.array(3.0),
        scale=2.0,
    )

    actual = IWLSWeights.score_squared("eta")(model, state)

    assert actual.shape == ()
    assert actual == pytest.approx(0.25)


def test_iwls_proposal_precision_uses_score_squared_weights():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    model, state, values = _gaussian_eta_liesel_state(
        eta=jnp.array([-1.0, 0.25, 1.5], dtype=Z.dtype),
        y=jnp.array([0.5, -0.75, 2.0], dtype=Z.dtype),
        scale=2.0,
        basis=Z,
        smooth_scale=state["tau"],
    )
    info = IWLSProposal(
        basis=Z,
        smooth_scale_name="tau",
        penalty=penalty,
        model=model,
        working_weights_fn=IWLSWeights.score_squared("eta"),
        basis_name="B",
    )
    expected_score = (values["y"] - values["eta"]) / values["scale"] ** 2
    expected_weights = jnp.clip(jnp.square(expected_score), min=1e-6)
    expected = _expected_precision(Z, penalty, expected_weights, values["tau"])

    assert jnp.allclose(info.precision(state), expected, rtol=1e-6, atol=1e-6)


def test_gaussian_iwls_weights_backwards_compatible_loc_alias():
    state = {"obs_scale": jnp.array([0.0, 2.0], dtype=jnp.float32)}

    actual = GaussianIWLSWeights.loc(scale_name="obs_scale")(DictModel(), state)
    expected = IWLSWeights.gaussian_loc(scale_name="obs_scale")(DictModel(), state)

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


def test_gaussian_iwls_weights_scale_returns_constant_two():
    _, _, state = _state(jnp.array(2.0, dtype=jnp.float32))

    actual = IWLSWeights.gaussian_scale()(DictModel(), state)

    assert actual.shape == ()
    assert actual == pytest.approx(2.0)


def test_gaussian_iwls_weights_backwards_compatible_scale_alias():
    _, _, state = _state(jnp.array(2.0, dtype=jnp.float32))

    actual = GaussianIWLSWeights.scale()(DictModel(), state)
    expected = IWLSWeights.gaussian_scale()(DictModel(), state)

    assert actual.shape == ()
    assert jnp.allclose(actual, expected)


def test_iwls_proposal_uses_supplied_working_weights_function():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    weights = jnp.array([0.5, 1.5, 2.5], dtype=Z.dtype)

    def working_weights(model, model_state):
        return model.extract_position(["custom_weights"], model_state)["custom_weights"]

    state = state | {"custom_weights": weights}
    info = IWLSProposal(
        basis=Z,
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=working_weights,
        basis_name="B",
    )

    expected = _expected_precision(Z, penalty, weights, state["tau"])

    assert jnp.allclose(info.working_weights(state), weights)
    assert jnp.allclose(info.precision(state), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("term", list(_structured_term_class_examples()))
def test_iwls_proposal_from_term_extracts_geometry_from_supported_terms(term):
    model = _model_for_term(term)

    proposal = IWLSProposal.from_term(term, IWLSWeights.gaussian_scale())
    expected_basis = (
        term.init_full_basis().value
        if isinstance(term, IndexingTerm)
        else term.basis.value
    )
    term_penalty = getattr(term, "_penalty", None)
    expected_penalty = (
        term_penalty.value
        if term_penalty is not None
        else jnp.eye(term.nbases, dtype=proposal.basis.dtype)
    )

    assert proposal.basis_name == term.basis.name
    assert jnp.allclose(proposal.basis, expected_basis)
    assert proposal.smooth_scale_name == term.scale.name
    assert jnp.allclose(proposal.penalty, expected_penalty)
    assert proposal.model is model
    assert proposal.scale_factored is False


def test_iwls_proposal_mcmc_spec_supports_indexing_term_subclasses():
    class CustomIndexingTerm(IndexingTerm):
        pass

    group = jnp.array([0, 2, 1, 2, 0])
    basis = Basis(group, xname="custom_group", penalty=None, use_callback=False)
    scale = lsl.Var.new_param(jnp.array(1.5), name="tau_custom_index")
    term = CustomIndexingTerm(
        basis,
        penalty=None,
        scale=scale,
        name="custom_index",
    )
    model = lsl.Model([term])
    spec = IWLSProposal.mcmc_spec(term, fallback_chol_info=None)
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)
    proposal = kernel.chol_info_fn.__self__
    expected_basis = jax.nn.one_hot(group, num_classes=term.nclusters)

    assert isinstance(proposal, IWLSProposal)
    assert proposal.basis_name == term.basis.name
    assert jnp.allclose(proposal.basis, expected_basis)
    assert jnp.allclose(proposal.penalty, jnp.eye(term.nclusters))
    assert kernel.chol_info_fn(model.state).shape == (term.nclusters, term.nclusters)


def test_iwls_proposal_precision_matches_explicit_one_hot_for_indexing_term():
    term, model, group, penalty = _indexing_term_and_model()
    weights = jnp.array([0.5, 1.5, 2.0, 0.75, 1.25], dtype=penalty.dtype)

    proposal = IWLSProposal.from_term(term, IWLSWeights.constant(weights))
    explicit_basis = jax.nn.one_hot(group, num_classes=term.nclusters)
    expected = _expected_precision(
        explicit_basis,
        penalty,
        weights,
        term.scale.value,
    )

    assert jnp.allclose(proposal.basis, explicit_basis)
    assert jnp.allclose(proposal.precision(model.state), expected)
    assert jnp.allclose(
        proposal.chol_info(model.state),
        jnp.linalg.cholesky(expected),
    )


def test_gaussian_iwls_proposals_can_be_constructed_from_indexing_term():
    term, model, group, penalty = _indexing_term_and_model()
    explicit_basis = jax.nn.one_hot(group, num_classes=term.nclusters)

    loc_proposal = GaussianLocIWLSProposal.from_term(term, scale_name="obs_scale")
    scale_proposal = GaussianScaleIWLSProposal.from_term(term)

    assert loc_proposal.basis_name == term.basis.name
    assert jnp.allclose(loc_proposal.basis, explicit_basis)
    assert jnp.allclose(loc_proposal.penalty, penalty)
    assert loc_proposal.model is model
    assert loc_proposal.chol_info(model.state).shape == (term.nclusters, term.nclusters)

    assert scale_proposal.basis_name == term.basis.name
    assert jnp.allclose(scale_proposal.basis, explicit_basis)
    assert jnp.allclose(scale_proposal.penalty, penalty)
    assert scale_proposal.model is model
    assert scale_proposal.chol_info(model.state).shape == (
        term.nclusters,
        term.nclusters,
    )


def test_gaussian_iwls_loc_spec_supports_indexing_term():
    term, model, group, _ = _indexing_term_and_model()
    explicit_basis = jax.nn.one_hot(group, num_classes=term.nclusters)

    spec = gaussian_iwls_spec_loc(
        term,
        scale_name="obs_scale",
        fallback_chol_info=None,
    )
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)
    proposal = kernel.chol_info_fn.__self__

    assert isinstance(proposal, GaussianLocIWLSProposal)
    assert jnp.allclose(proposal.basis, explicit_basis)
    assert kernel.chol_info_fn(model.state).shape == (term.nclusters, term.nclusters)


def test_gaussian_iwls_scale_spec_supports_indexing_term():
    term, model, group, _ = _indexing_term_and_model()
    explicit_basis = jax.nn.one_hot(group, num_classes=term.nclusters)

    spec = gaussian_iwls_spec_scale(term, fallback_chol_info=None)
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)
    proposal = kernel.chol_info_fn.__self__

    assert isinstance(proposal, GaussianScaleIWLSProposal)
    assert jnp.allclose(proposal.basis, explicit_basis)
    assert kernel.chol_info_fn(model.state).shape == (term.nclusters, term.nclusters)


def test_iwls_proposal_supports_riterm_with_unobserved_clusters():
    group = jnp.array([0, 1, 0, 1])
    basis = Basis(group, xname="ri_group", penalty=None, use_callback=False)
    scale = lsl.Var.new_param(jnp.array(1.5, dtype=jnp.float32), name="tau_ri_full")
    term = RITerm(basis, penalty=None, scale=scale, name="ri_full")
    term.mapping = CategoryMapping({"a": 0, "b": 1, "c": 2})
    term.labels = ["a", "b", "c"]
    term.coef.value = jnp.zeros(term.nclusters)

    model = lsl.Model([term])
    proposal = IWLSProposal.from_term(term, IWLSWeights.constant())
    explicit_basis = jax.nn.one_hot(group, num_classes=term.nclusters)
    expected = _expected_precision(
        explicit_basis,
        jnp.eye(term.nclusters),
        1.0,
        scale.value,
    )

    assert term.nclusters == 3
    assert proposal.basis.shape == (group.size, term.nclusters)
    assert jnp.allclose(proposal.basis, explicit_basis)
    assert jnp.allclose(proposal.penalty, jnp.eye(term.nclusters))
    assert jnp.allclose(proposal.precision(model.state), expected)


def test_iwls_proposal_from_term_rejects_terms_without_model():
    term, _ = _term_and_scale()

    with pytest.raises(ValueError, match="attached to a model"):
        IWLSProposal.from_term(term, IWLSWeights.gaussian_scale())


def test_iwls_proposal_from_term_rejects_anisotropic_terms():
    term = _interaction_term()

    with pytest.raises(NotImplementedError, match="isotropic penalties"):
        IWLSProposal.from_term(term, IWLSWeights.constant())


def test_gaussian_iwls_proposals_can_be_constructed_from_term():
    term, model = _term_model()

    loc_proposal = GaussianLocIWLSProposal.from_term(term, scale_name="obs_scale")
    scale_proposal = GaussianScaleIWLSProposal.from_term(term)

    assert loc_proposal.basis_name == term.basis.name
    assert jnp.allclose(loc_proposal.basis, term.basis.value)
    assert loc_proposal.smooth_name == term.name
    assert loc_proposal.smooth_scale_name == term.scale.name
    assert loc_proposal.scale_name == "obs_scale"
    assert loc_proposal.model is model
    assert loc_proposal.n == term.value.shape[0]

    assert scale_proposal.basis_name == term.basis.name
    assert jnp.allclose(scale_proposal.basis, term.basis.value)
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
        basis=Z,
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=working_weights,
        basis_name="B",
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


def test_iwls_proposal_kernel_factory_tunes_step_size_for_constant_weights():
    Z, penalty, state = _state(jnp.array(2.0, dtype=jnp.float32))
    proposal = IWLSProposal(
        basis=Z,
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=IWLSWeights.constant(),
        basis_name="B",
    )

    kernel = proposal.kernel_factory()(["coef"], fallback_chol_info=None)

    assert kernel.initial_step_size == pytest.approx(
        _iwls_kernel_default_initial_step_size()
    )
    assert kernel.da_tune_step_size is True
    assert kernel.fallback_chol_info is None
    assert jnp.allclose(
        kernel.chol_info_fn(state),
        jnp.linalg.cholesky(_expected_precision(Z, penalty, 1.0, state["tau"])),
    )


def test_iwls_proposal_kernel_factory_uses_iwls_step_default_when_tuning_requested():
    Z, penalty, _ = _state(jnp.array(2.0, dtype=jnp.float32))
    proposal = IWLSProposal(
        basis=Z,
        smooth_scale_name="tau",
        penalty=penalty,
        model=DictModel(),
        working_weights_fn=IWLSWeights.gaussian_scale(),
        basis_name="B",
    )

    kernel = proposal.kernel_factory()(["coef"], da_tune_step_size=True)

    assert kernel.initial_step_size == pytest.approx(
        _iwls_kernel_default_initial_step_size()
    )
    assert kernel.da_tune_step_size is True


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


def test_iwls_spec_builds_tuned_kernel_with_constant_unit_weights():
    term, model = _term_model()

    spec = IWLSProposal.mcmc_spec(term, fallback_chol_info=None)
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)
    proposal = kernel.chol_info_fn.__self__

    assert isinstance(spec, gs.MCMCSpec)
    assert spec.kernel_kwargs == {"term": term}
    assert isinstance(kernel, gs.IWLSKernel)
    assert kernel.position_keys == (term.coef.name,)
    assert kernel.initial_step_size == pytest.approx(
        _iwls_kernel_default_initial_step_size()
    )
    assert kernel.da_tune_step_size is True
    assert kernel.fallback_chol_info is None
    assert isinstance(proposal, IWLSProposal)
    assert not isinstance(proposal, GaussianLocIWLSProposal | GaussianScaleIWLSProposal)
    assert proposal.basis_name == term.basis.name
    assert proposal.smooth_scale_name == term.scale.name
    assert proposal.model is model
    assert proposal.scale_factored is False
    assert proposal.working_weights(model.state) == pytest.approx(1.0)
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)


def test_iwls_spec_forwards_kernel_kwargs():
    term, _ = _term_model()

    spec = IWLSProposal.mcmc_spec(
        term,
        initial_step_size=0.25,
        da_tune_step_size=True,
        fallback_chol_info="chol_of_modified_info",
    )
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert kernel.initial_step_size == pytest.approx(0.25)
    assert kernel.da_tune_step_size is True
    assert kernel.fallback_chol_info == "chol_of_modified_info"


def test_iwls_spec_can_disable_step_size_tuning_for_constant_weights():
    term, _ = _term_model()

    spec = IWLSProposal.mcmc_spec(
        term,
        initial_step_size=0.75,
        da_tune_step_size=False,
    )
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert kernel.initial_step_size == pytest.approx(0.75)
    assert kernel.da_tune_step_size is False


def test_iwls_spec_rejects_factored_terms():
    term, _ = _term_model(factor_scale=True)

    with pytest.raises(ValueError, match="scale-factored"):
        IWLSProposal.mcmc_spec(term)


def test_iwls_spec_rejects_anisotropic_terms():
    term = _interaction_term()

    with pytest.raises(NotImplementedError, match="isotropic penalties"):
        IWLSProposal.mcmc_spec(term)


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


def test_gaussian_iwls_spec_loc_uses_iwls_step_default_when_tuning_requested():
    term, _ = _term_model()

    spec = gaussian_iwls_spec_loc(
        term,
        scale_name="obs_scale",
        da_tune_step_size=True,
    )
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert kernel.initial_step_size == pytest.approx(
        _iwls_kernel_default_initial_step_size()
    )
    assert kernel.da_tune_step_size is True


def test_gaussian_iwls_spec_loc_rejects_factored_terms():
    term, _ = _term_model(factor_scale=True)

    with pytest.raises(ValueError, match="scale-factored"):
        gaussian_iwls_spec_loc(term, scale_name="obs_scale")


def test_gaussian_iwls_spec_loc_rejects_anisotropic_terms():
    term = _interaction_term()

    with pytest.raises(NotImplementedError, match="isotropic penalties"):
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


def test_gaussian_iwls_spec_scale_rejects_anisotropic_terms():
    term = _interaction_term()

    with pytest.raises(NotImplementedError, match="isotropic penalties"):
        gaussian_iwls_spec_scale(term)


def test_set_mcmc_specs_assigns_structured_terms_only():
    term, obs_scale = _term_and_scale()
    offset = lsl.Var.new_value(jnp.ones_like(term.value), name="offset")
    predictor = AdditivePredictor("loc", intercept=False)
    predictor += term, offset

    IWLSProposal.set_mcmc_specs(predictor, fallback_chol_info=None)
    model = lsl.Model([predictor, obs_scale])
    spec = term.coef.inference
    kernel = spec.kernel([term.coef.name], **spec.kernel_kwargs)

    assert isinstance(spec, gs.MCMCSpec)
    assert getattr(offset, "inference", None) is None
    assert isinstance(kernel.chol_info_fn.__self__, IWLSProposal)
    assert kernel.da_tune_step_size is True
    assert kernel.chol_info_fn.__self__.working_weights(model.state) == pytest.approx(
        1.0
    )
    assert kernel.chol_info_fn(model.state).shape == (term.nbases, term.nbases)


def test_set_mcmc_specs_skips_anisotropic_terms_with_verbose_log(caplog):
    term = _interaction_term()
    predictor = AdditivePredictor("loc", intercept=False)
    predictor += term

    with caplog.at_level(logging.INFO, logger="liesel_gam.iwls_proposals"):
        IWLSProposal.set_mcmc_specs(predictor, verbose=True)

    assert term.coef.inference is None
    assert "anisotropic IWLS not supported" in caplog.text
    assert "inference left unchanged" in caplog.text


def test_set_mcmc_specs_rejects_factored_terms():
    term, _ = _term_and_scale(factor_scale=True)
    predictor = AdditivePredictor("loc", intercept=False)
    predictor += term

    with pytest.raises(ValueError, match="scale-factored"):
        IWLSProposal.set_mcmc_specs(predictor)


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


def test_apply_gaussian_iwls_spec_loc_skips_anisotropic_terms():
    term = _interaction_term()
    predictor = AdditivePredictor("loc", intercept=False)
    predictor += term

    apply_gaussian_iwls_spec_loc(
        predictor,
        scale_name="obs_scale",
        fallback_chol_info=None,
    )

    assert term.coef.inference is None


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


def test_apply_gaussian_iwls_spec_scale_skips_anisotropic_terms():
    term = _interaction_term()
    predictor = AdditivePredictor("scale", intercept=False)
    predictor += term

    apply_gaussian_iwls_spec_scale(predictor, fallback_chol_info=None)

    assert term.coef.inference is None


def test_apply_gaussian_iwls_spec_scale_rejects_factored_terms():
    term, _ = _term_and_scale(factor_scale=True)
    predictor = AdditivePredictor("scale", intercept=False)
    predictor += term

    with pytest.raises(ValueError, match="scale-factored"):
        apply_gaussian_iwls_spec_scale(predictor)
