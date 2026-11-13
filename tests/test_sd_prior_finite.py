import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam

from .test_prior_penalty_rank import prior_rank


@pytest.mark.parametrize("factor_scale", [False, True])
def test_ri_calibrates_all_declared_categories_once(factor_scale) -> None:
    starts = []
    for labels, grid_size in ((["b", "a"], 2), (["b", "a", "a", "a"], 257)):
        prior = gam.SDPrior.from_max_abs_effect(
            2.0, probability=0.9, draws=512, grid_size=grid_size
        )
        group = gam.CatVar(labels, categories=["b", "unused", "a"], name="group")
        term = gam.TermBuilder.from_dict({}).ri(
            group, scale=prior, factor_scale=factor_scale
        )
        result = prior.calibrate(term.basis)
        assert term.scale.value == pytest.approx(result.initial_scales[0])
        assert result.diagnostics["evaluation_points"] == 3
        assert result.diagnostics["rank"] == 3
        assert term.coef.value.shape == (3,)
        assert term.basis.value.ndim == 1
        model = lsl.Model(term)
        term.coef.value = jnp.ones(3)
        model.update()
        expected = float(term.scale.value) if factor_scale else 1.0
        np.testing.assert_allclose(term.value, expected)
        starts.append(float(term.scale.value))
        assert np.isfinite(model.log_prob)
    assert starts[0] == starts[1]


@pytest.mark.parametrize(
    "settings, rank",
    [
        ({}, 3),
        ({"diagonal_penalty": False}, 3),
        ({"k": 3}, 2),
        ({"k": 3, "diagonal_penalty": False}, 2),
        ({"absorb_cons": False, "diagonal_penalty": False, "scale_penalty": False}, 3),
        ({"absorb_cons": False}, 3),
    ],
)
def test_mrf_calibrates_transformed_region_values(settings, rank) -> None:
    group = gam.CatVar(
        ["west", "east", "north", "west"],
        categories=["west", "unused", "east", "north"],
        name="region",
    )
    neighbors = {
        "east": ["north"],
        "north": ["east", "unused"],
        "unused": ["north", "west"],
        "west": ["unused"],
    }
    prior = gam.SDPrior.from_max_abs_effect(
        2.0, probability=0.9, grid_size=2, draws=4096
    )
    term = gam.TermBuilder.from_dict({}).mrf(
        group, nb=neighbors, scale=prior, **settings
    )
    result = prior.calibrate(term.basis)
    assert term.scale.value == pytest.approx(result.initial_scales[0])
    assert result.diagnostics["evaluation_points"] == 4
    assert result.diagnostics["rank"] == rank
    assert term.mapping is group.mapping
    model = lsl.Model(term)
    assert np.isfinite(model.log_prob)
    rng = np.random.default_rng(312)
    assert term.basis.penalty is not None
    covariance = np.linalg.pinv(
        np.asarray(term.basis.penalty.value, dtype=float), rcond=1e-5, hermitian=True
    )
    coefficients = rng.multivariate_normal(np.zeros(term.nbases), covariance, 30000)
    tau = np.sqrt(result.prior_scale) * rng.exponential(size=(30000, 1))
    prediction = model.predict(
        samples=lsl.Position({term.coef.name: jnp.asarray(coefficients * tau)}),
        predict=[term.name],
        newdata=lsl.Position({"region": ["west", "unused", "east", "north"]}),
    )[term.name]
    assert np.mean(np.max(np.abs(prediction), axis=-1) <= 2.0) == pytest.approx(
        0.9, abs=0.015
    )


@pytest.mark.parametrize("use_callback", [False, True])
@pytest.mark.parametrize("penalty", [None, np.diag([4.0, 0.0])])
def test_custom_f_uses_generic_scalar_matrix_geometry(use_callback, penalty) -> None:
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=1024)
    tb = gam.TermBuilder.from_dict(
        {"x": np.linspace(-1.0, 1.0, 20)}, default_scale_fn=prior
    )
    term = tb.f(
        "x",
        basis_fn=lambda x: jnp.column_stack((x[:, 0], x[:, 0] ** 2)),
        row_wise=True,
        use_callback=use_callback,
        penalty=penalty,
    )
    result = prior.calibrate(term.basis)
    assert result.diagnostics["evaluation_points"] == 129
    assert result.diagnostics["rank"] == (2 if penalty is None else 1)
    assert term.scale.value == pytest.approx(result.initial_scales[0])
    assert np.isfinite(lsl.Model(term).log_prob)


@pytest.mark.parametrize("factor_scale", [False, True])
@pytest.mark.parametrize(
    "penalty",
    [
        None,
        np.eye(3),
        np.diag([1.0, 4.0, 0.0]),
        np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_ri_fixed_penalties_follow_category_codes(penalty, factor_scale) -> None:
    group = gam.CatVar(
        ["last", "first"], categories=["last", "unused", "first"], name="g"
    )
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=4096)
    term = gam.TermBuilder.from_dict({}).ri(
        group, penalty=penalty, scale=prior, factor_scale=factor_scale
    )
    result = prior.calibrate(term.basis)
    rank = 2 if penalty is not None and penalty[-1, -1] == 0 else 3
    assert result.diagnostics["rank"] == rank
    assert term.coef.value.shape == (3,)
    model = lsl.Model(term)
    term.coef.value = jnp.array([1.0, 2.0, 3.0])
    model.update()
    expected = np.array([1.0, 3.0])
    if factor_scale:
        expected[0] *= float(term.scale.value)
        if rank == 3:
            expected[1] *= float(term.scale.value)
    np.testing.assert_allclose(term.value, expected, rtol=1e-6)
    assert np.isfinite(model.log_prob)

    # Independent prior simulation: the null-space coefficient is fixed at zero.
    rng = np.random.default_rng(691)
    covariance = np.eye(3) if penalty is None else np.linalg.pinv(penalty)
    tau = np.sqrt(result.prior_scale) * rng.exponential(size=(30000, 1))
    effects = rng.multivariate_normal(np.zeros(3), covariance, 30000) * tau
    assert np.mean(np.max(np.abs(effects), axis=1) <= 2.0) == pytest.approx(
        0.9, abs=0.015
    )


def test_ri_explicit_zero_penalty_is_not_iid() -> None:
    tb = gam.TermBuilder.from_dict({"g": ["b", "a"]})
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=64)
    with pytest.raises(ValueError, match="nonzero stochastic penalty"):
        tb.ri("g", scale=prior, penalty=np.zeros((2, 2)))


@pytest.mark.parametrize("row_wise", [None, False])
def test_custom_f_unknown_row_dependence_requires_an_explicit_scale(row_wise) -> None:
    tb = gam.TermBuilder.from_dict(
        {"x": np.linspace(-1, 1, 10)},
        default_scale_fn=gam.SDPrior.from_max_abs_effect(2.0, probability=0.9),
    )
    with pytest.raises(ValueError, match=r"f\(x\).*scalar row-wise.*explicit"):
        tb.f("x", basis_fn=lambda x: x, row_wise=row_wise)
    term = tb.f("x", basis_fn=lambda x: x, row_wise=row_wise, scale=1.0)
    assert np.isfinite(lsl.Model(term).log_prob)


@pytest.mark.parametrize("shape", ["vector", "tensor", "multiple_inputs"])
def test_custom_f_unsupported_automatic_geometry_fails_clearly(shape) -> None:
    tb = gam.TermBuilder.from_dict({"x": np.linspace(-1, 1, 10)})
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9)
    if shape == "multiple_inputs":
        with pytest.raises(ValueError, match="scalar row-wise"):
            tb.f("x", "x", basis_fn=lambda x: x, row_wise=True, scale=prior)
    else:
        fn = (lambda x: x[:, 0]) if shape == "vector" else (lambda x: x[:, :, None])
        with pytest.raises(ValueError, match="matrix-valued basis"):
            tb.f("x", basis_fn=fn, row_wise=True, scale=prior, use_callback=False)


@pytest.mark.parametrize("explicit_start", [None, 1.7])
@pytest.mark.parametrize("multiplier_kind", ["numeric", "random", "smooth"])
def test_rs_and_vc_preserve_coefficient_priors_and_starts(
    explicit_start, multiplier_kind
) -> None:
    prior = gam.SDPrior.from_max_abs_effect(
        2.0, probability=0.9, draws=256, value=explicit_start
    )
    tb = gam.TermBuilder.from_dict(
        {"x": np.linspace(-1.0, 1.0, 20), "g": ["b", "a"] * 10},
        default_scale_fn=prior,
    )
    smooth = tb.ps("x", k=6)
    original_scale = smooth.scale
    variance = original_scale.value_node[0]
    parameter = variance.bijected_var
    prior_node = parameter.dist_node
    parameter_start = np.array(parameter.value)
    scale_start = np.array(smooth.scale.value)
    if multiplier_kind == "random":
        multiplier = lsl.Var.new_param(
            jnp.linspace(-100.0, 100.0, 20),
            dist=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="multiplier",
        )
    elif multiplier_kind == "smooth":
        multiplier = smooth
    else:
        multiplier = "x"

    varying = tb.vc(smooth, by="x" if multiplier_kind == "smooth" else multiplier)
    slope = tb.rs(multiplier, cluster="g")
    group_term = slope.value_node["cluster"]
    assert isinstance(group_term, gam.RITerm)
    standalone = tb.ri("g")
    assert varying.value_node["by"] is smooth
    assert slope.value_node["x"] is (
        smooth
        if multiplier_kind == "smooth"
        else (
            multiplier
            if multiplier_kind == "random"
            else tb.registry.get_numeric_obs("x")
        )
    )
    assert group_term.scale.value == pytest.approx(float(standalone.scale.value))
    group_prior = (
        group_term.scale.value_node[0]
        .bijected_var.dist_node.init_dist()
        .distribution.distribution
    )
    standalone_prior = (
        standalone.scale.value_node[0]
        .bijected_var.dist_node.init_dist()
        .distribution.distribution
    )
    assert group_prior.scale == pytest.approx(float(standalone_prior.scale))
    assert smooth.scale is original_scale
    assert parameter.dist_node is prior_node
    np.testing.assert_array_equal(parameter.value, parameter_start)
    np.testing.assert_array_equal(smooth.scale.value, scale_start)
    if explicit_start is not None:
        assert smooth.scale.value == pytest.approx(explicit_start)
        assert group_term.scale.value == pytest.approx(explicit_start)
    assert np.isfinite(lsl.Model([varying, slope]).log_prob)


@pytest.mark.parametrize("use_default", [False, True])
@pytest.mark.parametrize("use_callback", [False, True])
def test_custom_f_categorical_domain_requires_an_explicit_scale(
    use_default, use_callback
) -> None:
    group = gam.CatVar(["b", "a"], categories=["b", "a", "unused"], name="group")
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=64)
    tb = gam.TermBuilder.from_dict(
        {}, default_scale_fn=prior if use_default else gam.VarIGPrior(1.0, 0.005)
    )

    def quadratic(x):
        return jnp.column_stack((x[:, 0], x[:, 0] ** 2))

    with pytest.raises(ValueError, match="numeric scalar row-wise"):
        tb.f(
            group,
            basis_fn=quadratic,
            row_wise=True,
            use_callback=use_callback,
            scale="default" if use_default else prior,
        )
    term = tb.f(
        group, basis_fn=quadratic, row_wise=True, use_callback=use_callback, scale=1.0
    )
    assert np.isfinite(lsl.Model(term).log_prob)
    with pytest.raises(ValueError, match="numeric scalar row-wise"):
        prior.calibrate(term.basis)


@pytest.mark.parametrize("row_wise", [None, False])
def test_categorical_calibration_requires_row_wise_basis(row_wise) -> None:
    group = gam.CatVar(["a", "a", "b"], name="group")
    basis = gam.Basis(
        group,
        basis_fn=lambda codes: codes[:, None] - codes.mean(),
        use_callback=False,
        row_wise=row_wise,
    )
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=64)

    with pytest.raises(ValueError, match="row_wise=True"):
        prior.calibrate(basis)


@pytest.mark.parametrize("bad_code", [-1, 2, 0.5])
def test_categorical_calibration_rejects_invalid_indexed_codes(bad_code) -> None:
    group = gam.CatVar(["a", "b"], name="group")
    basis = gam.Basis(
        group,
        basis_fn=lambda codes: jnp.full(codes.shape, bad_code),
        penalty=None,
        use_callback=False,
        row_wise=True,
    )
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=64)

    with pytest.raises(ValueError, match="each declared code"):
        prior.calibrate(basis)


@pytest.mark.parametrize("penalty", [None, jnp.eye(2)])
def test_categorical_calibration_rejects_many_to_one_indexing(penalty) -> None:
    group = gam.CatVar(["a", "b", "c", "d"], name="group")
    basis = gam.Basis(
        group,
        basis_fn=lambda codes: codes // 2,
        penalty=penalty,
        use_callback=False,
        row_wise=True,
    )
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=64)

    with pytest.raises(ValueError, match="each declared code"):
        prior.calibrate(basis)


def test_raw_chain_mrf_calibrates_every_penalized_direction() -> None:
    regions = [f"r{i:03d}" for i in range(400)]
    neighbors = {
        region: regions[max(i - 1, 0) : i] + regions[i + 1 : i + 2]
        for i, region in enumerate(regions)
    }
    builder = gam.TermBuilder.from_dict({"region": regions})
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=128)

    term = builder.mrf("region", nb=neighbors, diagonal_penalty=False, scale=prior)
    result = prior.calibrate(term.basis)

    assert result.diagnostics["rank"] == len(regions) - 1
    assert prior_rank(term) == result.diagnostics["rank"]
    assert term.scale.value == pytest.approx(result.initial_scales[0])


def test_calibration_uses_rank_of_replacement_penalty() -> None:
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0, 1, 20)})
    basis = builder.ps("x", k=8).basis
    basis.update_penalty(jnp.eye(basis.nbases))
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=128)

    assert prior.calibrate(basis).diagnostics["rank"] == basis.nbases


def test_calibration_rejects_stale_fitted_rank() -> None:
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0, 1, 20)})
    basis = builder.ps("x", k=8).basis
    assert basis.penalty is not None
    basis.penalty.value = jnp.eye(basis.nbases)
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=128)

    with pytest.raises(ValueError, match="fitted rank no longer matches"):
        prior.calibrate(basis)


@pytest.mark.parametrize(
    "method, settings, x",
    [
        ("cr", {"k": 100}, np.linspace(0, 1, 200)),
        ("ps", {"k": 80, "penalty_order": 4}, np.linspace(0, 1, 200)),
        (
            "ps",
            {"k": 40, "penalty_order": 3},
            np.random.default_rng(1).normal(size=200),
        ),
    ],
)
def test_unresolved_raw_penalty_recommends_diagonalization(method, settings, x) -> None:
    builder = gam.TermBuilder.from_dict({"x": x})
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=128)
    build = getattr(builder, method)

    with pytest.raises(ValueError, match="diagonal_penalty=True"):
        build("x", **settings, diagonal_penalty=False, scale=prior)

    term = build("x", **settings, diagonal_penalty=True, scale=prior)
    assert np.isfinite(term.scale.value)


def test_ri_uses_rank_of_large_supplied_chain_penalty() -> None:
    levels = [f"r{i:03d}" for i in range(400)]
    differences = np.diff(np.eye(len(levels)), axis=0)
    penalty = differences.T @ differences
    builder = gam.TermBuilder.from_dict({"region": levels})
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=128)

    term = builder.ri("region", penalty=penalty, scale=prior)
    result = prior.calibrate(term.basis)

    assert result.diagnostics["rank"] == len(levels) - 1
    assert prior_rank(term) == result.diagnostics["rank"]


def test_f_uses_rank_of_supplied_penalty() -> None:
    # This eigenvalue exceeds the precision floor but is below the old
    # tolerance, which grows with the number of coefficients.
    penalty = np.diag([1.0] * 15 + [1.5e-6])
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0, 1, 32)})
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=128)

    term = builder.f(
        "x",
        basis_fn=lambda x: jnp.cos(x * jnp.arange(16)),
        penalty=penalty,
        row_wise=True,
        scale=prior,
    )
    result = prior.calibrate(term.basis)

    assert result.diagnostics["rank"] == 16
    assert prior_rank(term) == 16
