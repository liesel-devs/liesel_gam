import jax
import jax.numpy as jnp
import liesel.model as lsl

import liesel_gam as gam
from liesel_gam.consolidate_bases import consolidate_bases
from liesel_gam.term import _factorized_tensor_dot


def _term(basis, name: str) -> gam.StrctTerm:
    return gam.StrctTerm.f(
        gam.Basis(
            basis,
            xname=name,
            penalty=jnp.eye(basis.shape[-1]),
            use_callback=False,
        ),
        scale=1.0,
    )


def _explicit_basis(*bases):
    return jax.vmap(lambda *rows: jnp.kron(jnp.kron(rows[0], rows[1]), rows[2]))(*bases)


def test_factorized_three_way_is_jittable_and_has_no_full_design_intermediate():
    n = 31
    bases = (
        jax.random.normal(jax.random.key(1), (n, 3)),
        jax.random.normal(jax.random.key(2), (n, 4)),
        jax.random.normal(jax.random.key(3), (n, 5)),
    )
    marginal_sizes = (3, 4, 5)
    ncoef = 3 * 4 * 5
    coef = jnp.linspace(-1.0, 1.0, ncoef)

    def evaluate(beta):
        return _factorized_tensor_dot(
            beta,
            bases,
            marginal_sizes=marginal_sizes,
            indexed=(False, False, False),
        )

    expected = _explicit_basis(*bases) @ coef
    assert jnp.allclose(jax.jit(evaluate)(coef), expected, atol=1e-5)

    jaxpr = jax.make_jaxpr(evaluate)(coef).jaxpr
    output_shapes = {
        tuple(var.aval.shape)
        for equation in jaxpr.eqns
        for var in equation.outvars
        if hasattr(var, "aval") and hasattr(var.aval, "shape")
    }
    assert (n, ncoef) not in output_shapes


def test_interaction_model_contains_only_marginal_design_matrices():
    n = 40
    terms = (
        _term(jnp.ones((n, 3)), "x"),
        _term(jnp.ones((n, 4)), "y"),
        _term(jnp.ones((n, 5)), "z"),
    )
    interaction = gam.StrctInteractionTerm(*terms)
    model = lsl.Model([interaction])

    assert interaction.nbases == 60
    assert not hasattr(interaction, "basis")
    graph_shapes = {
        tuple(var.value.shape)
        for var in model.vars.values()
        if hasattr(var.value, "shape")
    }
    assert (n, interaction.nbases) not in graph_shapes
    assert {(n, 3), (n, 4), (n, 5)} <= graph_shapes


def test_indexed_marginal_uses_gather_and_retains_its_penalty():
    n = 12
    indices = jnp.arange(n) % 3
    penalty = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    indexed = gam.IndexingTerm(
        gam.Basis(indices, xname="group", penalty=None, use_callback=False),
        penalty=penalty,
        scale=1.0,
    )
    dense = _term(jnp.column_stack((jnp.ones(n), jnp.linspace(0.0, 1.0, n))), "x")
    interaction = gam.StrctInteractionTerm(indexed, dense)
    coef = jnp.linspace(-1.0, 1.0, interaction.nbases)
    explicit = jax.vmap(jnp.kron)(
        jax.nn.one_hot(indices, indexed.nbases), dense.basis.value
    )

    interaction.coef.value = coef
    interaction.update()
    assert jnp.allclose(interaction.value, explicit @ coef, atol=1e-5)
    assert jnp.allclose(interaction.penalties[0], penalty)
    assert interaction.marginal_bases[0].value.shape == (n,)


def test_multivariate_three_way_matches_explicit_basis():
    n = 9
    bases = (
        jax.random.normal(jax.random.key(4), (n, 2)),
        jax.random.normal(jax.random.key(5), (n, 3)),
        jax.random.normal(jax.random.key(6), (n, 4)),
    )
    term = gam.MultivariateStrctInteractionTerm.f(
        *(_term(basis, name) for basis, name in zip(bases, ("x", "y", "z"))),
        dimension_penalties=[jnp.eye(2)],
        dimension_scales=[1.0],
    )
    coef = jnp.linspace(-1.0, 1.0, term.coef.value.size)
    explicit = _explicit_basis(*bases)

    term.coef.value = coef
    term.latent.update()
    expected = explicit @ coef.reshape(term.nbases, term.latent_ndim)
    assert not hasattr(term, "basis")
    assert jnp.allclose(term.latent.value, expected, atol=1e-5)


def test_consolidation_keeps_only_marginal_bases():
    n = 15
    x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, n), name="x")
    y = lsl.Var.new_obs(jnp.linspace(1.0, 2.0, n), name="y")
    bx = gam.Basis(
        x,
        basis_fn=lambda value: jnp.column_stack((jnp.ones_like(value), value)),
        penalty=jnp.eye(2),
        use_callback=False,
    )
    by = gam.Basis(
        y,
        basis_fn=lambda value: jnp.column_stack((jnp.ones_like(value), value)),
        penalty=jnp.eye(2),
        use_callback=False,
    )
    interaction = gam.StrctInteractionTerm(
        gam.StrctTerm.f(bx, scale=1.0),
        gam.StrctTerm.f(by, scale=1.0),
    )
    model, bases_model = consolidate_bases(lsl.Model([interaction]), copy=True)

    assert model.vars[bx.name].strong
    assert model.vars[by.name].strong
    assert set(bases_model.vars) >= {bx.name, by.name, "x", "y"}
    assert all(
        tuple(var.value.shape) != (n, interaction.nbases)
        for var in model.vars.values()
        if hasattr(var.value, "shape")
    )
