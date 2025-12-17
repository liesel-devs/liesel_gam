from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self, cast

import liesel.goose as gs
import liesel.model as lsl

from .var import UserVar

Array = Any

term_types = lsl.Var


class AdditivePredictor(UserVar):
    """
    A Liesel :class:`~liesel.model.Var` that represents an additive predictor.

    This is a special variable that allows you to add other Liesel varibales using the
    ``+=`` syntax (see examples).

    Parameters
    ----------
    name
        Name of the predictor variable.
    inv_link
        Inverse link function. If supplied, variables are added on the *link* level,
        and the predictor variable's value will be ``inv_link(sum(*inputs))``, where
        ``*inputs`` refers to the additive terms in this predictor.
    intercept
        Whether this predictor should be initialized with an intercept. You can supply
        booleans, or a :class:`liesel.model.Var`. In the latter case, this var is
        taken as the intercept. The default intercept has a constant prior.
    intercept_name
        Name of the automatically created intercept variable (if ``intercept=True``).
        If this name contains the placeholder ``{subscript}``, it will be filled with
        the predictor name to create a unique intercept name for this predictor.

    Examples
    --------
    Basic example:

    >>> import liesel_gam as gam
    >>> import liesel.model as lsl

    >>> loc = gam.AdditivePredictor("loc")

    Now we add a variable using the ``+`` syntax. The value of the predictor is the
    sum of the values of its inputs.
    >>> loc += lsl.Var.new_value(1.0, name="s(x)")
    >>> loc.value
    1.0

    The input terms can be accessed:

    >>> loc.terms
    {'s(x)': Var(name="s(x)")}

    This term got initialized with a default intercept, which has no distribution
    node (corresponding to a constant prior).

    >>> loc.intercept
    Var(name="$\\beta_{0,loc}$")
    >>> print(loc.intercept.dist_node)
    None

    After adding a second term, the value of the predictor is updated:

    >>> loc += lsl.Var.new_value(2.5, name="s(x2)")
    >>> loc.terms
    {'s(x)': Var(name="s(x)"), 's(x2)': Var(name="s(x2)")}
    >>> loc.value
    3.5

    Using an inverse link function:

    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import liesel_gam as gam
    >>> scale = gam.AdditivePredictor("scale", inv_link=jnp.exp)
    >>> scale += lsl.Var.new_value(1.0, name="s(x)")
    >>> scale.terms
    {'s(x)': Var(name="s(x)")}
    >>> scale.intercept
    Var(name="$\\beta_{0,scale}$")
    >>> scale.value
    Array(2.7182817, dtype=float32, weak_type=True)

    Using a custom intercept:

    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> import liesel_gam as gam
    >>> intercept_var = lsl.Var.new_param(
    ...     value=3.0,
    ...     distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=10.0),
    ...     name="b0",
    ... )
    >>> loc = gam.AdditivePredictor("loc", intercept=intercept_var)
    >>> loc.intercept
    Var(name="b0")
    >>> loc.value
    3.0
    >>> loc.intercept.dist_node
    Dist(name="b0_log_prob")

    Multiple terms can be added at the same time:

    >>> import liesel.model as lsl
    >>> import liesel_gam as gam
    >>> loc = gam.AdditivePredictor("loc")
    >>> sx1 = lsl.Var.new_value(1.0, name="s(x1)")
    >>> sx2 = lsl.Var.new_value(1.0, name="s(x2)")
    >>> loc += sx1, sx2
    >>> loc.terms
    {'s(x1)': Var(name="s(x1)"), 's(x2)': Var(name="s(x2)")}
    """

    def __init__(
        self,
        name: str,
        inv_link: Callable[[Array], Array] | None = None,
        intercept: bool | lsl.Var = True,
        intercept_name: str = "$\\beta{subscript}$",
    ) -> None:
        if inv_link is None:

            def inv_link(x):
                return x

        def _sum(*args, intercept, **kwargs):
            # the + 0. implicitly ensures correct dtype also for empty predictors
            return inv_link(sum(args) + sum(kwargs.values()) + 0.0 + intercept)

        if intercept and not isinstance(intercept, lsl.Var):
            name_cleaned = name.replace("$", "")

            intercept_: lsl.Var | float = lsl.Var.new_param(
                name=intercept_name.format(subscript="_{0," + name_cleaned + "}"),
                value=0.0,
                distribution=None,
                inference=gs.MCMCSpec(gs.IWLSKernel.untuned),
            )
        elif isinstance(intercept, lsl.Var):
            intercept_ = intercept
        else:
            intercept_ = 0.0

        super().__init__(lsl.Calc(_sum, intercept=intercept_), name=name)
        self.update()
        self.terms: dict[str, term_types] = {}
        """Dictionary of terms in this predictor."""

    @property
    def intercept(self) -> lsl.Var | lsl.Node:
        """This term's intercept."""
        return self.value_node["intercept"]

    @intercept.setter
    def intercept(self, value: lsl.Var | lsl.Node):
        self.value_node["intercept"] = value

    def update(self) -> Self:
        return cast(Self, super().update())

    def __iadd__(self, other: term_types | Sequence[term_types]) -> Self:
        if isinstance(other, term_types):
            self.append(other)
        else:
            self.extend(other)
        return self

    def append(self, term: term_types) -> None:
        """
        Appends a single term to this predictor's sum inputs.

        Examples
        --------
        >>> import liesel.model as lsl
        >>> import liesel_gam as gam
        >>> loc = gam.AdditivePredictor("loc")
        >>> sx1 = lsl.Var.new_value(1.0, name="s(x)")
        >>> loc.append(sx1)
        >>> loc.terms
        {'s(x)': Var(name="s(x)")}
        """
        if not isinstance(term, term_types):
            raise TypeError(f"{term} is of unsupported type {type(term)}.")

        if term.name in self.terms:
            raise RuntimeError(f"{self} already contains a term of name {term.name}.")

        self.value_node.add_inputs(term)
        self.terms[term.name] = term
        self.update()

    def extend(self, terms: Sequence[term_types]) -> None:
        """
        Appends a sequence of terms to this predictor's sum inputs.

        Examples
        --------
        >>> import liesel.model as lsl
        >>> import liesel_gam as gam
        >>> loc = gam.AdditivePredictor("loc")
        >>> sx1 = lsl.Var.new_value(1.0, name="s(x1)")
        >>> sx2 = lsl.Var.new_value(1.0, name="s(x2)")
        >>> loc.extend([sx1, sx2])
        >>> loc.terms
        {'s(x1)': Var(name="s(x1)"), 's(x2)': Var(name="s(x2)")}
        """
        for term in terms:
            self.append(term)

    def __getitem__(self, name) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"
