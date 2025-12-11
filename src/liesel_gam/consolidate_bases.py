"""
Instances of :class:`.Basis` may use non-jittable basis functions.
In batched optimization, this may lead to inefficient repeated basis evaluation.
If the basis functions depend on ryp for interfacing to R, which many do, this will
not only be inefficient but fail completely, because R is not thread-safe.

To solve these issues, this module provides utility functions to create models that
can be safely and efficiently used in batched operations:

- :func:`.consolidate_bases` splits a model into the model, where
   all bases are turned into strong, observed varibales, and a model for the bases.
   The former can be used in batched optimization, the latter can be used to still
   conveniently evaluate all relevant bases based on their original inputs.

- :func:`.evaluate_bases` takes a position of input data, evaluate the corresponding
  bases in the provided model, and returns a position of the evaluated bases.
"""

import liesel.model as lsl
from liesel.goose.types import Position

from .basis import Basis


def _remove_singleton_vars(gb: lsl.GraphBuilder) -> lsl.GraphBuilder:
    """
    Removes all singleton variables from the provided GraphBuilder.
    """
    model = gb.build_model()

    G = model.var_graph
    singletons1 = [n for n, d in G.degree() if d == 0]
    singletons2 = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    singleton_vars = set(singletons1 + singletons2)

    G = model.node_graph
    singletons1 = [n for n, d in G.degree() if d == 0]
    singletons2 = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    singleton_nodes = set(singletons1 + singletons2)

    nodes, vars_ = model.pop_nodes_and_vars()

    for var in singleton_vars:
        vars_.pop(var.name, None)

    for node in singleton_nodes:
        nodes.pop(node.name, None)

    gb = lsl.GraphBuilder(to_float32=model._to_float32)
    gb.add(*vars_.values())
    return gb


def consolidate_bases(
    model: lsl.Model, copy: bool = True
) -> tuple[lsl.Model, lsl.Model]:
    """
    Turns all :class:`.Basis` variables in the provided model into strong,
    observed :class:`liesel.model.Var` variables.

    Returns a new model that depends only on the strong bases, and a model that
    holds the original bases and their input data.

    If ``copy=False``, all data will be extracted from the original model, instead
    of creating copies. This saves memory, but renders the original model empty.
    """
    if copy:
        nodes, vars_ = model.copy_nodes_and_vars()
    else:
        nodes, vars_ = model.pop_nodes_and_vars()

    gb = lsl.GraphBuilder(to_float32=model._to_float32)
    gb.add(*nodes.values(), *vars_.values())

    weak_bases = []

    for var in gb.vars:
        if not isinstance(var, Basis):
            continue
        weak_basis = var
        strong_basis = lsl.Var.new_obs(weak_basis.update().value, name=weak_basis.name)
        gb.replace_var(old=weak_basis, new=strong_basis)
        weak_bases.append(weak_basis)

    gb = _remove_singleton_vars(gb)

    bases_model = (
        lsl.GraphBuilder(to_float32=model._to_float32).add(*weak_bases).build_model()
    )
    model = gb.build_model()

    return model, bases_model


def evaluate_bases(newdata: Position, model: lsl.Model) -> Position:
    """
    Evaluates all :class:`.Basis` variables in the provided model at the provided
    newdata position.
    """
    state = model.update_state(newdata)

    basis_names = []
    for var in model.vars.values():
        if isinstance(var, Basis):
            basis_names.append(var.name)

    return Position(model.extract_position(basis_names, state))
