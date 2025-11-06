from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self, cast

import liesel.goose as gs
import liesel.model as lsl

from .var import BasisDot, Term, UserVar

Array = Any

term_types = Term | BasisDot


class AdditivePredictor(UserVar):
    def __init__(
        self,
        name: str,
        inv_link: Callable[[Array], Array] | None = None,
        intercept: bool | lsl.Var = True,
    ) -> None:
        if inv_link is None:

            def inv_link(x):
                x

        def _sum(*args, intercept, **kwargs):
            # the + 0. implicitly ensures correct dtype also for empty predictors
            return inv_link(sum(args) + sum(kwargs.values()) + 0.0 + intercept)

        if intercept and not isinstance(intercept, lsl.Var):
            intercept = lsl.Var.new_param(
                name=f"{name}_intercept",
                value=0.0,
                distribution=None,
                inference=gs.MCMCSpec(gs.IWLSKernel),
            )
        else:
            intercept = 0.0

        super().__init__(lsl.Calc(_sum, intercept=intercept), name=name)
        self.update()
        self.terms: dict[str, term_types] = {}
        """Dictionary of terms in this predictor."""

    @property
    def intercept(self) -> lsl.Var | float:
        return self.value_node["intercept"]

    @intercept.setter
    def intercept(self, value: lsl.Var):
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
        if not isinstance(term, term_types):
            raise TypeError(f"{term} is of unsupported type {type(term)}.")

        if term.name in self.terms:
            raise RuntimeError(f"{self} already contains a term of name {term.name}.")

        self.value_node.add_inputs(term)
        self.terms[term.name] = term
        self.update()

    def extend(self, terms: Sequence[term_types]) -> None:
        for term in terms:
            self.append(term)

    def __getitem__(self, name) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"
