import jax
import jax.numpy as jnp
import pytest
from liesel.contrib.splines import equidistant_knots as kn

import liesel_gam as gam
from liesel_gam.experimental.approx_bspline import BSplineApprox, basis_matrix


class TestBSplineApprox:
    def test_init(self):
        x = jax.random.uniform(jax.random.key(1234), shape=(40,))
        knots = kn(x, n_param=20)

        bspline = BSplineApprox(knots)
        assert bspline.basis_grid.shape == (1000, 20)

    @pytest.mark.parametrize("seed", (1, 2, 3, 4, 5))
    def test_approx_equal(self, seed):
        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(40,))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(20,))

        basis = basis_matrix(x, knots)

        bspline = BSplineApprox(knots, ngrid=2000)
        rmse = jnp.sqrt(jnp.sum(((basis @ coef) - bspline.dot(x, coef)) ** 2))
        assert rmse < 0.1

    @pytest.mark.parametrize("seed", (1, 2, 3, 4, 5))
    def test_rmse_shrinks(self, seed):
        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(40,))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(20,))

        basis = basis_matrix(x, knots)

        bspline = BSplineApprox(knots, ngrid=1000)
        rmse1000 = jnp.sqrt(jnp.sum(((basis @ coef) - bspline.dot(x, coef)) ** 2))

        bspline = BSplineApprox(knots, ngrid=2000)
        rmse2000 = jnp.sqrt(jnp.sum(((basis @ coef) - bspline.dot(x, coef)) ** 2))

        assert rmse1000 > rmse2000

        bspline = BSplineApprox(knots, ngrid=10000)
        rmse10000 = jnp.sqrt(jnp.sum(((basis @ coef) - bspline.dot(x, coef)) ** 2))

        assert rmse2000 > rmse10000

    def test_scalar_x(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(40,))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(20,))

        bspline = BSplineApprox(knots, ngrid=1000)

        assert jnp.allclose(bspline.dot(x[0], coef), bspline.dot(x[:1], coef))

    def test_jit(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(40,))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(20,))

        bspline = BSplineApprox(knots, ngrid=1000)

        fx = jax.jit(bspline.dot)(x, coef)
        assert not jnp.any(jnp.isnan(fx))

        fx, fxd = jax.jit(bspline.dot_and_deriv)(x, coef)
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

    def test_batch_coef(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(40,))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(4, 100, 20))

        bspline = BSplineApprox(knots, ngrid=1000)

        fx = jax.jit(bspline.dot)(x, coef)
        assert fx.shape == (4, 100, 40)
        assert not jnp.any(jnp.isnan(fx))

        fx, fxd = jax.jit(bspline.dot_and_deriv)(x, coef)
        assert fx.shape == (4, 100, 40)
        assert fxd.shape == (4, 100, 40)
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

    def test_batch_x(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(4, 100, 40))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(20))

        bspline = BSplineApprox(knots, ngrid=1000)

        fx = jax.jit(bspline.dot)(x, coef)
        assert fx.shape == (4, 100, 40)
        assert not jnp.any(jnp.isnan(fx))

        fx, fxd = jax.jit(bspline.dot_and_deriv)(x, coef)
        assert fx.shape == (4, 100, 40)
        assert fxd.shape == (4, 100, 40)
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

    def test_batch_both(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(4, 100, 40))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(4, 100, 20))

        bspline = BSplineApprox(knots, ngrid=1000)

        fx = jax.jit(bspline.dot)(x, coef)
        assert fx.shape == (4, 100, 40)
        assert not jnp.any(jnp.isnan(fx))

        fx, fxd = jax.jit(bspline.dot_and_deriv)(x, coef)
        assert fx.shape == (4, 100, 40)
        assert fxd.shape == (4, 100, 40)
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

    def test_batch_both_nomatch(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(3, 100, 40))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(4, 100, 20))

        bspline = BSplineApprox(knots, ngrid=1000)

        with pytest.raises(ValueError):
            jax.jit(bspline.dot)(x, coef)

        with pytest.raises(ValueError):
            jax.jit(bspline.dot_and_deriv)(x, coef)

    def test_grad(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(40))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(20))

        bspline = BSplineApprox(knots, ngrid=1000)

        def dotsum(x, coef):
            return bspline.dot(x, coef).sum()

        grad_x = jax.grad(dotsum, argnums=0)(x, coef)
        assert not jnp.any(jnp.isnan(grad_x))
        assert grad_x.shape == (40,)

        grad_coef = jax.grad(dotsum, argnums=1)(x, coef)
        assert not jnp.any(jnp.isnan(grad_coef))
        assert grad_coef.shape == (20,)

    def test_grad_dot_and_deriv(self):
        seed = 1

        key = jax.random.key(seed)
        k0, k1 = jax.random.split(key)
        x = jax.random.uniform(k0, shape=(40))
        knots = kn(x, n_param=20)

        coef = jax.random.normal(k1, shape=(20))

        bspline = BSplineApprox(knots, ngrid=1000)

        def dotsum(x, coef):
            dot, deriv = bspline.dot_and_deriv(x, coef)
            return dot.sum() + deriv.sum()

        grad_x = jax.grad(dotsum, argnums=0)(x, coef)
        assert not jnp.any(jnp.isnan(grad_x))
        assert grad_x.shape == (40,)

        grad_coef = jax.grad(dotsum, argnums=1)(x, coef)
        assert not jnp.any(jnp.isnan(grad_coef))
        assert grad_coef.shape == (20,)

    def test_z(self):
        nbases = 20

        x = jax.random.uniform(jax.random.key(1234), shape=(40,))
        coef = jax.random.normal(jax.random.key(4321), shape=((nbases - 1),))
        knots = kn(x, n_param=nbases)

        basis = basis_matrix(x, knots)
        Z = gam.LinearConstraintEVD.sumzero_term(basis)

        bspline = BSplineApprox(knots, Z=Z)
        bspline.dot(x, coef)
