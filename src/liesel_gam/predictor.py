from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self, cast

import liesel.model as lsl

from .var import Term, UserVar

Array = Any


class AdditivePredictor(UserVar):
    def __init__(
        self, name: str, inv_link: Callable[[Array], Array] | None = None
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
        self.terms: dict[str, Term] = {}
        """Dictionary of terms in this predictor."""

    def update(self) -> Self:
        return cast(Self, super().update())

    def __iadd__(self, other: Term | Sequence[Term]) -> Self:
        if isinstance(other, Term):
            self.append(other)
        else:
            self.extend(other)
        return self

    def append(self, term: Term) -> None:
        if not isinstance(term, Term):
            raise TypeError(f"{term} is of unsupported type {type(term)}.")

        if term.name in self.terms:
            raise RuntimeError(f"{self} already contains a term of name {term.name}.")

        if term.includes_intercept and self.includes_intercept:
            raise ValueError(
                f"{term.includes_intercept=}, but {self.includes_intercept=}. "
                "Cannot add a second intercept."
            )

        self.value_node.add_inputs(term)
        self.terms[term.name] = term
        self.update()

    def extend(self, terms: Sequence[Term]) -> None:
        for term in terms:
            self.append(term)

    @property
    def includes_intercept(self) -> bool | None:
        any_intercept_none: bool = False
        intercept_found = False
        for term in self.terms.values():
            if term.includes_intercept is None:
                any_intercept_none = True

            if term.includes_intercept:
                intercept_found = True
                return intercept_found  # early return

        if not intercept_found and any_intercept_none:
            return None

        return False

    def __getitem__(self, name) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"
