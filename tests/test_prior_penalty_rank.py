"""
The coefficient prior's penalty rank.

The prior density contains ``-rank * log(tau)``. These tests read the rank from
that density, so they check what the model computes rather than how it stores the
rank.
"""

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pytest

import liesel_gam as gam


def prior_rank(term: gam.StrctTerm) -> int:
    """Rank in the coefficient prior, read from its density at ``tau = e``."""
    assert term.coef.dist_node is not None
    dist = term.coef.dist_node.init_dist().copy(scale=jnp.exp(1.0))
    log_prob = dist.log_prob(jnp.zeros(term.nbases))
    return round(-float(log_prob))


def chain_neighbors(n: int) -> dict[str, list[str]]:
    labels = [str(i) for i in range(n)]
    return {
        labels[i]: [labels[j] for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)
    }


@pytest.fixture(scope="module")
def x() -> np.ndarray:
    return np.linspace(0.0, 1.0, 200)


@pytest.fixture(scope="module")
def regions() -> np.ndarray:
    # 400 regions in a chain. With diagonal_penalty=False, matrix_rank counts 398
    # of the 399 penalized directions left after the sum-to-zero constraint.
    return np.array([str(i) for i in range(400)])


class TestPriorUsesKnownRank:
    def test_raw_chain_mrf(self, regions) -> None:
        tb = gam.TermBuilder.from_dict({"r": regions})
        term = tb.mrf("r", nb=chain_neighbors(400), diagonal_penalty=False, scale=1.0)

        assert term.nbases == 399
        assert prior_rank(term) == 399

    def test_raw_cr(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.cr("x", k=40, diagonal_penalty=False, scale=1.0)

        # 39 coefficients after the sum-to-zero constraint; the linear trend is
        # unpenalized.
        assert term.nbases == 39
        assert prior_rank(term) == 38

    def test_raw_cr_matches_diagonalized(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        raw = tb.cr("x", k=40, diagonal_penalty=False, scale=1.0)
        diagonal = tb.cr("x", k=40, diagonal_penalty=True, scale=1.0)

        assert prior_rank(raw) == prior_rank(diagonal)

    def test_diagonalized_np_keeps_rank(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.np("x", k=100, scale=1.0)

        assert term.nbases == 98
        assert prior_rank(term) == 98

    def test_updated_penalty_uses_its_own_rank(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        basis = tb.bases.cr("x", k=10, diagonal_penalty=False)
        basis.update_penalty(jnp.eye(basis.nbases))

        term = gam.StrctTerm.f(basis, scale=1.0)

        assert term.nbases == 9
        assert prior_rank(term) == 9

    def test_other_penalty_than_basis_penalty(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        basis = tb.bases.cr("x", k=10, diagonal_penalty=False)

        term = gam.StrctTerm(basis, penalty=jnp.eye(basis.nbases), scale=1.0)

        assert prior_rank(term) == 9

    def test_explicit_rank_in_mvn_structured_prior(self) -> None:
        scale = lsl.Var.new_value(jnp.exp(1.0))
        penalty = lsl.Value(jnp.eye(3))
        prior = gam.term.mvn_structured_prior(scale, penalty, penalty_rank=2)

        log_prob = prior.init_dist().log_prob(jnp.zeros(3))

        assert jnp.allclose(log_prob, -2.0)

    def test_invalid_rank_in_mvn_structured_prior(self) -> None:
        scale = lsl.Var.new_value(1.0)
        penalty = lsl.Value(jnp.eye(3))

        with pytest.raises(ValueError, match="penalty_rank"):
            gam.term.mvn_structured_prior(scale, penalty, penalty_rank=4)


class TestSuppliedPenaltyRank:
    def test_ri_with_float32_penalty_cast_to_float64(self) -> None:
        regions = [str(i) for i in range(50)]
        differences = np.diff(np.eye(len(regions), dtype=np.float32), n=2, axis=0)
        penalty = ((differences.T @ differences) / np.float32(3)).astype(np.float64)
        tb = gam.TermBuilder.from_dict({"region": regions})

        term = tb.ri("region", penalty=penalty)

        assert prior_rank(term) == 47

    def test_f_with_constrained_penalty(self) -> None:
        tb = gam.TermBuilder.from_dict({"x": np.array([-1.0, -0.5, 0.5, 1.0])})

        term = tb.f(
            "x",
            basis_fn=lambda x: jnp.concatenate([x, jnp.ones_like(x), x**3], axis=1),
            penalty=jnp.diag(jnp.array([0.0, 1.0, 1.0])),
            row_wise=True,
        ).constrain("sumzero_term")

        assert prior_rank(term) == 1


class TestTermTransformsUpdateRank:
    def test_constrain(self) -> None:
        x = jnp.linspace(0.0, 1.0, 4)
        basis = gam.Basis(
            jnp.column_stack([jnp.ones_like(x), x]), xname="x", penalty=jnp.eye(2)
        )
        term = gam.StrctTerm.f(basis, scale=1.0).constrain(jnp.ones((1, 2)))

        assert term.nbases == 1
        assert prior_rank(term) == 1

    def test_diagonalize_raw_cr(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        basis = tb.bases.cr("x", k=40, diagonal_penalty=False)
        term = gam.StrctTerm.f(basis, scale=1.0).diagonalize_penalty()

        assert prior_rank(term) == 38


class TestFactorScale:
    def test_raw_full_rank_mrf(self, regions) -> None:
        # After the sum-to-zero constraint, the raw chain MRF penalty has full
        # rank, so the scale can be factored out without diagonalizing.
        tb = gam.TermBuilder.from_dict({"r": regions})
        term = tb.mrf(
            "r",
            nb=chain_neighbors(400),
            diagonal_penalty=False,
            scale=1.0,
            factor_scale=True,
        )

        assert term.scale_is_factored


class TestGibbsKernelUsesPriorRank:
    def test_raw_cr(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.cr("x", k=40, diagonal_penalty=False)
        assert isinstance(term.scale, gam.ScaleIG)
        model = lsl.Model([term])

        variance = term.scale.value_node[0]
        kernel = gam.init_star_ig_gibbs(
            position_keys=[variance.name], coef=term.coef, scale=term.scale
        )
        key = jax.random.key(1)
        draw = kernel._transition_fn(key, model.state)[variance.name]

        # With zero coefficients, the full conditional is
        # InverseGamma(a + rank / 2, b).
        a = variance.dist_node["concentration"].value
        b = variance.dist_node["scale"].value
        expected = b / jax.random.gamma(key, a + 0.5 * 38)
        assert jnp.allclose(draw, expected)

    def test_factored_raw_mrf(self, regions) -> None:
        tb = gam.TermBuilder.from_dict({"r": regions})
        term = tb.mrf(
            "r", nb=chain_neighbors(400), diagonal_penalty=False, factor_scale=True
        )
        assert isinstance(term.scale, gam.ScaleIG)
        model = lsl.Model([term])

        variance = term.scale.value_node[0]
        spec = variance.inference
        assert isinstance(spec, gs.MCMCSpec)
        kernel = spec.kernel([variance.name], **spec.kernel_kwargs)
        key = jax.random.key(1)
        draw = kernel._transition_fn(key, model.state)[variance.name]

        a = variance.dist_node["concentration"].value
        b = variance.dist_node["scale"].value
        expected = b / jax.random.gamma(key, a + 0.5 * 399)
        assert jnp.allclose(draw, expected)


class TestPriorRankFitsStoredPenalty:
    """
    smoothcon reports the float64 rank. The prior samples with a float32 ``eigh``
    of the stored penalty, so it only counts directions that copy resolves.
    """

    def test_raw_ill_conditioned_ps_caps_prior_rank(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.ps("x", k=80, penalty_order=4, diagonal_penalty=False, scale=1.0)

        # The basis keeps the float64 rank; the prior counts fewer directions.
        assert term.basis._penalty_rank == 76
        assert prior_rank(term) < 76

    def test_raw_ill_conditioned_ps_samples_are_finite(self) -> None:
        x = np.linspace(0.0, 1.0, 400)
        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.ps("x", k=200, penalty_order=4, diagonal_penalty=False, scale=1.0)

        assert term.coef.dist_node is not None
        dist = term.coef.dist_node.init_dist()
        samples = dist.sample(3, seed=jax.random.key(0))

        assert bool(jnp.all(jnp.isfinite(samples)))

    def test_diagonalized_ps_keeps_full_rank(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.ps("x", k=80, penalty_order=4, scale=1.0)

        assert prior_rank(term) == 76

    def test_raw_low_rank_mrf_keeps_full_rank(self, regions) -> None:
        # The low-rank MRF penalty is diagonal, so float32 holds every entry.
        tb = gam.TermBuilder.from_dict({"r": regions})
        term = tb.mrf(
            "r", nb=chain_neighbors(400), k=60, diagonal_penalty=False, scale=1.0
        )

        assert term.basis._penalty_rank is not None
        assert prior_rank(term) == term.basis._penalty_rank


class TestNpUsesFloat64Setup:
    def test_np_geometry_matches_float64_reference(self, x) -> None:
        import smoothcon

        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.np("x", k=100, scale=1.0)
        design = np.asarray(term.basis.value, dtype=np.float64)
        # The same knots as the term, which computes them in float32.
        knots = np.asarray(
            smoothcon.pspline(jnp.asarray(x), k=100, degree=3, penalty_order=2).knots,
            dtype=np.float64,
        )

        with jax.enable_x64(True):
            values = jnp.asarray(np.asarray(x, dtype=np.float32), dtype=jnp.float64)
            reference = (
                smoothcon.pspline(values, k=100, degree=3, penalty_order=2, knots=knots)
                .constrain("constant_and_linear", values=values)
                .scale_penalty(values=values)
                .diagonalize_penalty(values=values)
            )
            expected = np.asarray(reference.basis(values))

        rank = reference.rank
        assert term.nbases == 98
        assert prior_rank(term) == rank == 98
        covariance = design[:, :rank] @ design[:, :rank].T
        expected_covariance = expected[:, :rank] @ expected[:, :rank].T
        error = np.linalg.norm(covariance - expected_covariance) / np.linalg.norm(
            expected_covariance
        )
        assert error < 1e-5

    def test_np_records_its_constraint(self, x) -> None:
        tb = gam.TermBuilder.from_dict({"x": x})
        term = tb.np("x", k=20, scale=1.0)

        assert term.basis.constraint == "constant_and_linear"
