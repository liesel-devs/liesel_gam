from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self, cast

import liesel.goose as gs
import liesel.model as lsl

from .var import Intercept, Term, UserVar

Array = Any

term_types = Term | Intercept


class AdditivePredictor(UserVar):
    def __init__(
        self,
        name: str,
        inv_link: Callable[[Array], Array] | None = None,
        add_intercept: bool = True,
    ) -> None:
        if inv_link is None:

            def _sum(*args, **kwargs):
                # the + 0. implicitly ensures correct dtype also for empty predictors
                return sum(args) + sum(kwargs.values()) + 0.0
        else:

            def _sum(*args, **kwargs):
                # the + 0. implicitly ensures correct dtype also for empty predictors
                return inv_link(sum(args) + sum(kwargs.values()) + 0.0)

        super().__init__(lsl.Calc(_sum), name=name)
        self.update()
        self.terms: dict[str, term_types] = {}
        """Dictionary of terms in this predictor."""

        if add_intercept:
            self += Intercept(
                name=f"{name}_intercept",
                value=0.0,
                distribution=None,
                inference=gs.MCMCSpec(gs.IWLSKernel),
            )

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
