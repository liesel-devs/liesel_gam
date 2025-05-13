from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self, cast

import liesel.model as lsl

Array = Any


class AdditivePredictor(lsl.Var):
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
        self.terms: dict[str, lsl.Var] = {}
        """Dictionary of terms in this predictor."""

    def update(self) -> Self:
        return cast(Self, super().update())

    def __add__(self, other: lsl.Var) -> Self:
        self.value_node.add_inputs(other)
        self.terms[other.name] = other
        return self.update()

    def __iadd__(self, other: lsl.Var) -> Self:
        self.value_node.add_inputs(other)
        self.terms[other.name] = other
        return self.update()

    def __getitem__(self, name) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"
