from functools import reduce

import jax
import jax.numpy as jnp
import jax.random as jrd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.contrib import splines
from liesel.distributions import MultivariateNormalDegenerate as MVND
from tensorflow_probability.substrates.jax import tf2jax as tf

import liesel_gam as gam
from liesel_gam import dist as gd


class TestMultivariateNormalSingular:
    def test_log_prob(self) -> None:
        pen = splines.pspline_penalty(d=10, diff=2)
        mvns = gam.MultivariateNormalSingular(
            loc=0.0, scale=1.0, penalty=pen, penalty_rank=8
        )
        mvnd = MVND.from_penalty(loc=0.0, var=1.0, pen=pen)

        x = jrd.normal(jrd.key(1), (20, 10))

        assert mvns.log_prob(x).shape == (20,)

        impl_diffs = jnp.diff(mvnd.log_prob(x) - mvns.log_prob(x))
        assert jnp.allclose(impl_diffs, 0.0, atol=1e-5)

    def test_event_shape(self) -> None:
        pen = splines.pspline_penalty(d=10, diff=2)
        mvns = gam.MultivariateNormalSingular(
            loc=0.0, scale=1.0, penalty=pen, penalty_rank=8
        )

        assert mvns.event_shape == tf.TensorShape([10])
        assert mvns.event_shape_tensor() == 10
        assert jnp.allclose(mvns._event_shape_tensor(), mvns.event_shape_tensor())

    def test_samples(self) -> None:
        pen = splines.pspline_penalty(d=10, diff=2)
        mvns = gam.MultivariateNormalSingular(
            loc=0.0, scale=1.0, penalty=pen, penalty_rank=8
        )
        mvnd = MVND.from_penalty(loc=0.0, var=1.0, pen=pen)

        dist = tfd.MultivariateNormalFullCovariance(
            loc=0.0, covariance_matrix=jnp.eye(10)
        )

        dist.sample((1,), seed=jrd.key(1))

        x1 = mvns.sample((1,), seed=jrd.key(1))
        x2 = mvnd.sample((1,), seed=jrd.key(1))
        assert jnp.allclose(x1, x2)


def pspline_penalty(nparam: int, random_walk_order: int = 2):
    """
    Builds an (nparam x nparam) P-spline penalty matrix.
    """
    D = jnp.diff(jnp.identity(nparam), random_walk_order, axis=0)
    return D.T @ D


class TestStructuredPenaltyOperator:
    def test_materialize_2d(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)

        evd1 = jnp.linalg.eigh(K1)
        evd2 = jnp.linalg.eigh(K2)

        scales = jnp.array([1.0, 2.0])

        op = gd.StructuredPenaltyOperator(
            scales=scales,
            penalties=[K1, K2],
            penalties_eigvalues=[evd1.eigenvalues, evd2.eigenvalues],
        )

        Ka = op.materialize_precision()

        I1 = jnp.eye(K1.shape[-1])
        I2 = jnp.eye(K2.shape[-1])

        tau21 = scales[0] ** 2
        tau22 = scales[1] ** 2

        Kb = jnp.kron(K1, I2) / tau21 + jnp.kron(I1, K2) / tau22

        assert jnp.allclose(Ka, Kb)

    def test_materialize_3d(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        evd1 = jnp.linalg.eigh(K1)
        evd2 = jnp.linalg.eigh(K2)
        evd3 = jnp.linalg.eigh(K3)

        scales = jnp.array([1.0, 2.0, 0.5])

        op = gd.StructuredPenaltyOperator(
            scales=scales,
            penalties=[K1, K2, K3],
            penalties_eigvalues=[evd1.eigenvalues, evd2.eigenvalues, evd3.eigenvalues],
        )

        Ka = op.materialize_precision()

        I1 = jnp.eye(K1.shape[-1])
        I2 = jnp.eye(K2.shape[-1])
        I3 = jnp.eye(K3.shape[-1])

        tau21 = scales[0] ** 2
        tau22 = scales[1] ** 2
        tau23 = scales[2] ** 2

        Kb = (
            jnp.kron(jnp.kron(K1, I2), I3) / tau21
            + jnp.kron(jnp.kron(I1, K2), I3) / tau22
            + jnp.kron(jnp.kron(I1, I2), K3) / tau23
        )

        assert jnp.allclose(Ka, Kb)

    def test_log_pdet_is_correct(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        scales = jnp.array([1.0, 2.0, 0.5])

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales,
            penalties=[K1, K2, K3],
        )

        ldet1 = op.log_pdet()

        K = op.materialize_precision()
        eig = jnp.linalg.eigh(K)
        mask = eig.eigenvalues > 1e-5
        ldet2 = jnp.log(eig.eigenvalues[mask]).sum()

        assert jnp.allclose(ldet1, ldet2)

    def test_compute_masks(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        scales = jnp.array([1.0, 2.0, 0.5])

        with pytest.raises(ValueError):
            gd.StructuredPenaltyOperator.from_penalties(
                scales=scales, penalties=[K1, K2, K3], eps=1.0
            )

    def test_validate_penalties(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        scales = jnp.array([1.0, 2.0, 0.5])

        gd.StructuredPenaltyOperator.from_penalties(
            scales=scales, penalties=[K1, K2, K3], validate_args=True
        )

        with pytest.raises(ValueError):
            gd.StructuredPenaltyOperator.from_penalties(
                scales=scales, penalties=[K1, K2], validate_args=True
            )

    def test_materialize_penalty(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        scales = jnp.array([1.0, 2.0, 0.5])

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales, penalties=[K1, K2, K3]
        )

        K = op.materialize_penalty()

        p = len(op._penalties)

        dims = [op._penalties[j].shape[-1] for j in range(p)]

        def kron_all(mats):
            """Kronecker product of a list of matrices."""
            return reduce(jnp.kron, mats)

        def one_batch(variances, *penalties):
            # Build K_1 / tau1^2 as initial value
            factors = [penalties[0] if i == 0 else jnp.eye(dims[i]) for i in range(p)]

            K = kron_all(factors) / variances[0]

            # Add remaining K_j / tau_j^2
            for j in range(1, p):
                factors = [
                    penalties[j] if i == j else jnp.eye(dims[i]) for i in range(p)
                ]
                Kj = kron_all(factors)
                K = K + Kj / variances[j]

            return K

        Kb = one_batch(jnp.ones(p), *[K1, K2, K3])

        assert jnp.allclose(K, Kb)
        assert not jnp.allclose(K, op.materialize_precision())

    def test_log_pdet_is_jittable(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        scales = jnp.array([1.0, 2.0, 0.5])

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales,
            penalties=[K1, K2, K3],
        )

        masks = op._masks
        pens = op._penalties
        evals = op._penalties_eigvalues

        ldet1 = op.log_pdet()

        def ldet(scales):
            op = gd.StructuredPenaltyOperator(
                scales=scales,
                penalties=pens,
                penalties_eigvalues=evals,
                masks=masks,
            )
            return op.log_pdet()

        assert jnp.allclose(jax.jit(ldet)(scales), ldet1)

    def test_log_pdet_is_jittable_without_precomputed_mask(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        scales = jnp.array([1.0, 2.0, 0.5])

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales,
            penalties=[K1, K2, K3],
        )

        pens = op._penalties
        evals = op._penalties_eigvalues

        ldet1 = op.log_pdet()
        masks = op._masks

        def ldet(scales):
            op = gd.StructuredPenaltyOperator(
                scales=scales,
                penalties=pens,
                penalties_eigvalues=evals,
                masks=masks,
            )
            return op.log_pdet()

        assert jnp.allclose(jax.jit(ldet)(scales), ldet1)

    def test_log_pdet_grad_works(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        scales = jnp.array([1.0, 2.0, 0.5])

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales,
            penalties=[K1, K2, K3],
        )

        masks = op._masks
        pens = op._penalties
        evals = op._penalties_eigvalues

        def ldet(scales):
            op = gd.StructuredPenaltyOperator(
                scales=scales,
                penalties=pens,
                penalties_eigvalues=evals,
                masks=masks,
            )
            return op.log_pdet()

        assert not jnp.any(jnp.isnan(jax.grad(ldet)(scales)))

    def test_log_pdet_batched_one_batching_dim(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        K1 = jnp.stack((K1, K1), axis=0)
        K2 = jnp.stack((K2, K2), axis=0)
        K3 = jnp.stack((K3, K3), axis=0)

        scales = jnp.array([1.0, 2.0, 0.5])
        scales = jnp.stack((scales, scales), axis=0)

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales,
            penalties=[K1, K2, K3],
        )

        assert op.log_pdet().shape == (2,)

    def test_log_pdet_batched_two_batching_dims(self):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        K1 = jnp.stack((K1, K1), axis=0)
        K2 = jnp.stack((K2, K2), axis=0)
        K3 = jnp.stack((K3, K3), axis=0)

        K1 = jnp.stack((K1, K1, K1), axis=0)
        K2 = jnp.stack((K2, K2, K2), axis=0)
        K3 = jnp.stack((K3, K3, K3), axis=0)

        scales = jnp.array([1.0, 2.0, 0.5])
        scales = jnp.stack((scales, scales), axis=0)
        scales = jnp.stack((scales, scales, scales), axis=0)

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales,
            penalties=[K1, K2, K3],
        )

        batch_shape = (3, 2)

        assert op.log_pdet().shape == batch_shape

    @pytest.mark.parametrize("seed", (0, 1, 2, 3))
    def test_quad_form(self, seed):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)

        evd1 = jnp.linalg.eigh(K1)
        evd2 = jnp.linalg.eigh(K2)

        scales = jnp.array([1.0, 2.0])

        op = gd.StructuredPenaltyOperator(
            scales=scales,
            penalties=[K1, K2],
            penalties_eigvalues=[evd1.eigenvalues, evd2.eigenvalues],
        )

        K = op.materialize_precision()

        n = K.shape[-1]
        x = jax.random.normal(jax.random.key(seed), (n,))

        xKx = x.T @ K @ x
        xKx2 = op.quad_form(x)

        assert jnp.allclose(xKx, xKx2)

    @pytest.mark.parametrize("seed", (0, 1, 2, 3))
    def test_quad_form_batched(self, seed):
        K1 = pspline_penalty(6)
        K2 = pspline_penalty(8)
        K3 = pspline_penalty(5)

        K1 = jnp.stack((K1, K1), axis=0)
        K2 = jnp.stack((K2, K2), axis=0)
        K3 = jnp.stack((K3, K3), axis=0)

        scales = jnp.array([1.0, 2.0, 0.5])
        scales = jnp.stack((scales, scales), axis=0)

        op = gd.StructuredPenaltyOperator.from_penalties(
            scales=scales,
            penalties=[K1, K2, K3],
        )

        K = op.materialize_precision()

        n = K.shape[-1]
        x = jax.random.normal(jax.random.key(seed), (n,))

        def quad_form(x, K):
            xKx = x.T @ K @ x
            return xKx

        xKx = jax.vmap(quad_form, (None, 0))(x, K)
        xKx2 = op.quad_form(x)

        assert jnp.allclose(xKx, xKx2)

        x = jax.random.normal(jax.random.key(seed), K.shape[:-2] + (n,))

        xKx = jax.vmap(quad_form, (0, 0))(x, K)
        xKx2 = op.quad_form(x)

        assert jnp.allclose(xKx, xKx2)

        x = jax.random.normal(jax.random.key(seed), (3,) + K.shape[:-2] + (n,))

        with pytest.raises(ValueError):
            op.quad_form(x)


class TestMultivariateNormalStructuredSingular:
    def test_sample(self):
        K1 = pspline_penalty(6)[:-2, :-2]
        K2 = pspline_penalty(8)[:-2, :-2]

        n = K1.shape[-1] * K2.shape[-1]

        loc = jnp.zeros(n)
        scales = jnp.array([1.0, 2.0])
        dist = gd.MultivariateNormalStructured.from_penalties(
            loc=loc, scales=scales, penalties=[K1, K2]
        )

        x = dist.sample(seed=jax.random.key(1))
        assert jnp.allclose(jnp.asarray(x.shape), dist.event_shape_tensor())
        assert jnp.allclose(jnp.asarray(x.shape), dist._event_shape_tensor())

        x = dist.sample(5, seed=jax.random.key(1))
        assert x.shape == (5, n)

        x = dist.sample((5, 10), seed=jax.random.key(1))
        assert x.shape == (5, 10, n)

        dist = gd.MultivariateNormalStructured.from_penalties(
            loc=loc, scales=scales, penalties=[K1, K2], precompute_masks=False
        )

        with pytest.raises(ValueError):
            dist.sample(seed=jax.random.key(1))

    def test_from_penalties(self):
        K1 = pspline_penalty(6)[:-2, :-2]
        K2 = pspline_penalty(8)[:-2, :-2]

        fn = gd.MultivariateNormalStructured.get_locscale_constructor
        dist_constr = fn((K1, K2))

        n = K1.shape[-1] * K2.shape[-1]

        loc = jnp.zeros(n)
        scales = jnp.array([1.0, 2.0])
        d1 = dist_constr(loc=loc, scales=scales)
        d2 = gd.MultivariateNormalStructured.from_penalties(
            loc=loc, scales=scales, penalties=[K1, K2]
        )

        x = jax.random.normal(jax.random.key(1), (n,))

        assert jnp.allclose(d1.log_prob(x), d2.log_prob(x))

    def test_validate_penalties(self):
        K1 = pspline_penalty(6)[:-2, :-2]
        K2 = pspline_penalty(8)[:-2, :-2]

        fn = gd.MultivariateNormalStructured.get_locscale_constructor
        dist_constr = fn((K1, K2), validate_args=True)

        n = K1.shape[-1] * K2.shape[-1]

        loc = jnp.zeros(n)
        scales = jnp.array([1.0, 2.0])
        dist_constr(loc=loc, scales=scales)

        with pytest.raises(ValueError):
            dist_constr(loc=loc, scales=jnp.array([1.0]))

        with pytest.raises(ValueError):
            dist_constr(loc=jnp.zeros(3), scales=scales)

    @pytest.mark.parametrize("seed", (0, 1, 2, 3))
    def test_log_prob(self, seed):
        K1 = pspline_penalty(6)[:-2, :-2]
        K2 = pspline_penalty(8)[:-2, :-2]

        fn = gd.MultivariateNormalStructured.get_locscale_constructor
        dist_constr = fn((K1, K2))

        n = K1.shape[-1] * K2.shape[-1]

        loc = jnp.zeros(n)
        scales = jnp.array([1.0, 2.0])
        dist = dist_constr(loc=loc, scales=scales)

        K = dist._op.materialize_precision()

        x = jax.random.normal(jax.random.key(seed), (n,))

        lp1 = dist.log_prob(x)

        lp1 = lp1

        Ki = jnp.linalg.inv(K)
        dist2 = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=Ki)

        lp2 = dist2.log_prob(x)

        assert jnp.allclose(lp1, lp2, atol=1e-4)

    @pytest.mark.parametrize("seed", (0, 1, 2, 3))
    def test_log_prob_unnormalized(self, seed):
        K1 = pspline_penalty(6)[:-2, :-2]
        K2 = pspline_penalty(8)[:-2, :-2]

        fn = gd.MultivariateNormalStructured.get_locscale_constructor
        dist_constr_norm = fn((K1, K2), include_normalizing_constant=True)
        dist_constr_unnorm = fn((K1, K2), include_normalizing_constant=False)

        n = K1.shape[-1] * K2.shape[-1]

        loc = jnp.zeros(n)
        scales = jnp.array([1.0, 2.0])
        dist_norm = dist_constr_norm(loc=loc, scales=scales)
        dist_unnorm = dist_constr_unnorm(loc=loc, scales=scales)

        x1 = jax.random.normal(jax.random.key(seed), (n,))
        x2 = jax.random.normal(jax.random.key(seed + 1), (n,))

        lp1 = dist_norm.log_prob(x1)
        lp2 = dist_norm.log_prob(x2)

        lp1_unnorm = dist_unnorm.log_prob(x1)
        lp2_unnorm = dist_unnorm.log_prob(x2)

        assert lp1 != pytest.approx(lp1_unnorm)
        assert (lp1 - lp1_unnorm) == pytest.approx(lp2 - lp2_unnorm)

    def test_log_prob_no_precomputed_mask(self):
        seed = 1

        K1 = pspline_penalty(6)[:-2, :-2]
        K2 = pspline_penalty(8)[:-2, :-2]

        fn = gd.MultivariateNormalStructured.get_locscale_constructor
        dist_constr = fn((K1, K2), precompute_masks=False)

        n = K1.shape[-1] * K2.shape[-1]

        loc = jnp.zeros(n)
        scales = jnp.array([1.0, 2.0])
        dist = dist_constr(loc=loc, scales=scales)

        K = dist._op.materialize_precision()

        x = jax.random.normal(jax.random.key(seed), (n,))

        lp1 = dist.log_prob(x)

        lp1 = lp1

        Ki = jnp.linalg.inv(K)
        dist2 = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=Ki)

        lp2 = dist2.log_prob(x)

        assert jnp.allclose(lp1, lp2, atol=1e-4)

    def test_batching(self):
        k = jax.random.key(0)
        key_x1, key_x2, key_beta = jax.random.split(k, 3)

        B = 3

        tau21 = jnp.arange(B) + 0.5
        tau22 = 1 + jnp.arange(B) * 0.2

        K1 = tfd.Normal(0.0, 1.0).sample((B, 10, 10), key_x1)
        K1 = jax.vmap(lambda K: K @ K.T + jnp.eye(10))(K1)
        K2 = tfd.Normal(0.0, 1.0).sample((B, 20, 20), key_x2)
        K2 = jax.vmap(lambda K: K @ K.T + jnp.eye(20))(K2)

        eigenvalues1 = jax.vmap(jnp.linalg.eigvalsh)(K1)
        eigenvalues2 = jax.vmap(jnp.linalg.eigvalsh)(K2)

        K_tau2 = jax.vmap(
            lambda tau21, tau22, K1, K2: tau21 * jnp.kron(K1, jnp.eye(K2.shape[0]))
            + tau22 * jnp.kron(jnp.eye(K1.shape[0]), K2),
            (0, 0, 0, 0),
        )(tau21, tau22, K1, K2)

        op = gd.StructuredPenaltyOperator(
            jnp.c_[1 / jnp.sqrt(tau21), 1 / jnp.sqrt(tau22)],
            [K1, K2],
            [eigenvalues1, eigenvalues2],
        )

        beta = tfd.Normal(0.0, 1.0).sample((B, K_tau2.shape[-1]), key_beta)
        dist = gd.MultivariateNormalStructured(jnp.zeros_like(beta), op)

        assert jnp.allclose(
            jnp.asarray(dist.batch_shape), jnp.asarray(K_tau2.shape[:-2])
        )

        assert jnp.allclose(
            jnp.asarray(dist.batch_shape_tensor()), jnp.asarray(K_tau2.shape[:-2])
        )

        assert jnp.allclose(
            jnp.asarray(dist._batch_shape_tensor()), jnp.asarray(K_tau2.shape[:-2])
        )

        assert dist.log_prob(beta[0]).shape == (3,)
        assert dist.log_prob(beta).shape == (3,)

        dist = gd.MultivariateNormalStructured(jnp.zeros_like(beta[0]), op)

        assert dist.log_prob(beta[0]).shape == (3,)
