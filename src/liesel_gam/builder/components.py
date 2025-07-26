from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Self

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
from liesel.contrib import splines

from ..var import Basis, Intercept, LinearTerm, SmoothTerm, Term
from .registry import VariableRegistry


class FormulaComponent(ABC):
    """Base class for formula components."""

    @classmethod
    @abstractmethod
    def from_formula(cls, func_name: str, args: list, kwargs: dict, term: str) -> Self:
        """Create a component from a parsed formula string."""
        pass

    @abstractmethod
    def formula_key(self, simplified: bool) -> str:
        """Return a string how this component appears in a formula."""
        pass

    @abstractmethod
    def create_terms(
        self, registry: VariableRegistry, name_prefix: str = ""
    ) -> list[Term]:
        """Create terms from the requested variables."""
        pass

    @property
    @abstractmethod
    def includes_intercept(self) -> bool:
        """Whether this component includes an intercept term."""
        pass

    @property
    @abstractmethod
    def depends_on_vars(self) -> set[str]:
        """Return set of variable names this component depends on from dataframe."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.formula_key(simplified=False)})"

    def __str__(self):
        return self.formula_key(simplified=True)


class InterceptComponent(FormulaComponent):
    """Intercept component for formulas."""

    def __init__(self, has_intercept: bool = True):
        self._has_intercept = has_intercept

    @classmethod
    def from_formula(
        cls, func_name: str, args: list, kwargs: dict, term: str
    ) -> InterceptComponent:
        """Create an InterceptComponent (not used for intercept parsing)."""
        raise NotImplementedError(
            "InterceptComponent.from_formula should not be called"
        )

    def formula_key(self, simplified: bool) -> str:
        return "Intercept" if self._has_intercept else "NoIntercept"

    def create_terms(
        self, registry: VariableRegistry, name_prefix: str = ""
    ) -> list[Term]:
        if self._has_intercept:
            name = f"{name_prefix}_intercept" if name_prefix else "intercept"
            return [Intercept(name=name, inference=gs.MCMCSpec(gs.IWLSKernel))]
        else:
            return []

    @property
    def includes_intercept(self) -> bool:
        return self._has_intercept

    @property
    def depends_on_vars(self) -> set[str]:
        return set()


class LinearComponent(FormulaComponent):
    """Linear component for formulas."""

    def __init__(self, vars: Iterable[str], intercept: bool, name: str | None = None):
        self._intercept = intercept
        self._vars = vars
        self._name = name

    @classmethod
    def from_formula(
        cls, func_name: str, args: list, kwargs: dict, term: str
    ) -> LinearComponent:
        if len(args) != 1:
            raise ValueError(
                f"{func_name}() expects exactly one argument, got {len(args)}: {args}"
            )

        vars = []
        intercept: bool | None = None
        for arg in str(args[0]).strip().split("+"):
            arg = arg.strip()
            match arg:
                case "1":
                    if intercept is not None:
                        raise ValueError(f"unclear intercept spec in {args[0]}")
                    intercept = True
                case "0":
                    if intercept is not None:
                        raise ValueError(f"unclear intercept spec in {args[0]}")
                    intercept = False
                case _:
                    vars.append(arg)

        if "intercept" in kwargs and intercept is not None:
            raise ValueError(
                f"Cannot specify intercept both in args and kwargs: {args[0]}"
            )

        final_intercept: bool = kwargs.pop("intercept", False)
        name = kwargs.pop("name", None)

        if kwargs:
            raise ValueError(
                f"{func_name}() does not accept additional arguments: {kwargs}"
            )

        return LinearComponent(vars=vars, intercept=final_intercept, name=name)

    def formula_key(self, simplified: bool) -> str:
        inner = ""
        match self._intercept:
            case True:
                inner = "1"
            case False:
                inner = "0"

        if self._vars:
            inner += "+"

        inner += "+".join(self._vars)
        if not simplified and self._name:
            inner += f", name={self._name}"

        return f"linear({inner})"

    def create_terms(
        self, registry: VariableRegistry, name_prefix: str = ""
    ) -> list[Term]:
        if self._name:
            term_name = self._name
        else:
            base_name = self.formula_key(simplified=True)
            term_name = f"{name_prefix}_{base_name}" if name_prefix else base_name

        if not self._vars:
            # No variables, just intercept
            if self.includes_intercept:
                return [Intercept(name=term_name)]
            else:
                return []

        vars = [registry.get_numeric_var(var) for var in self._vars]

        def stack_terms(*vars_input):
            return jnp.column_stack([v for v in vars_input])

        stacked = lsl.Var.new_calc(stack_terms, *vars)

        intercept: bool = self.includes_intercept
        lt = LinearTerm(
            stacked,
            name=term_name,
            add_intercept=intercept,
            inference=gs.MCMCSpec(gs.IWLSKernel),
        )
        return [lt]

    @property
    def includes_intercept(self) -> bool:
        if self._intercept is None:
            return False
        else:
            return self._intercept

    @property
    def depends_on_vars(self) -> set[str]:
        return set(self._vars)

    @classmethod
    def merge_linear_components(
        cls,
        components: list[LinearComponent],
    ) -> LinearComponent:
        if not components:
            raise ValueError("Cannot merge an empty list of components")
        elif len(components) == 1:
            return components[0]
        else:
            var_names: set[str] = set()
            intercept_counter = 0
            no_intercept_counter = 0
            for comp in components:
                if comp._name:
                    raise ValueError(
                        "Cannot merge components with names."
                        f"found {comp.formula_key(False)}."
                    )
                match comp._intercept:
                    case True:
                        intercept_counter += 1
                    case False:
                        no_intercept_counter += 1
                    case None:
                        pass
                var_names.update(comp._vars)

            if intercept_counter > 0 and no_intercept_counter > 0:
                raise ValueError("Inconsistent intercept specifications")

            if intercept_counter > 0:
                intercept = True
            elif no_intercept_counter > 0:
                intercept = False
            else:
                raise RuntimeError("Should never be reached.")

            merged = LinearComponent(
                vars=list(var_names), intercept=intercept, name=None
            )
            return merged


class LieselSplineComponent(FormulaComponent):
    """Liesel spline component for ls() and similar spline function calls."""

    def __init__(self, func_name: str, args: list, kwargs: dict, term: str):
        if len(args) != 1:
            raise ValueError(
                f"{func_name}() expects exactly one variable, got {len(args)}: {args}"
            )

        self.func_name = func_name
        self.variable_name = args[0]
        self._term = term

        # pop liesel-specific parameters (not for splines)
        kwargs_copy = kwargs.copy()
        self.term_name = kwargs_copy.pop("name", None)
        self.ig_concentration = kwargs_copy.pop("ig_concentration", 0.01)
        self.ig_scale = kwargs_copy.pop("ig_scale", 0.01)

        # remaining kwargs are for splines
        self.spline_kwargs = kwargs_copy

        # validate splines parameters
        bs = self.spline_kwargs.get("bs", "ps")
        if bs != "ps":
            raise ValueError(f"{func_name}() only supports bs='ps', got bs='{bs}'")

    @classmethod
    def from_formula(
        cls, func_name: str, args: list, kwargs: dict, term: str
    ) -> LieselSplineComponent:
        """Create a LieselSplineComponent"""
        return LieselSplineComponent(func_name, args, kwargs, term)

    def formula_key(self, simplified: bool) -> str:
        if simplified:
            return f"{self.func_name}({self.variable_name})"
        else:
            return self._term

    def create_terms(
        self, registry: VariableRegistry, name_prefix: str = ""
    ) -> list[Term]:
        if self.term_name:
            term_name = self.term_name  # ignore prefix for named components
        else:
            base_name = self.formula_key(simplified=True)
            term_name = f"{name_prefix}_{base_name}" if name_prefix else base_name

        # get the input variable
        input_var = registry.get_numeric_var(self.variable_name)
        input_values = input_var.value

        # get spline parameters
        k = self.spline_kwargs.get("k", 10)

        # create spline basis
        knots = splines.equidistant_knots(input_values, n_param=k, order=3)
        penalty = splines.pspline_penalty(d=k, diff=2)

        basis = Basis(
            value=input_var,
            basis_fn=splines.basis_matrix,
            knots=knots,
            name=f"{term_name}_basis",
        )

        # create smooth term
        smooth_term = SmoothTerm.new_ig(
            basis=basis,
            penalty=penalty,
            name=term_name,
            inference=gs.MCMCSpec(gs.IWLSKernel),
            ig_concentration=self.ig_concentration,
            ig_scale=self.ig_scale,
        )
        return [smooth_term]

    @property
    def includes_intercept(self) -> bool:
        # Liesel splines do always include an intercept
        return True

    @property
    def depends_on_vars(self) -> set[str]:
        return {self.variable_name}


class FactorComponent(FormulaComponent):
    """Factor component for categorical variables in formulas."""

    def __init__(
        self,
        var_name: str,
        reference_level: str | None = None,
        contrasts: str = "treatment",
        name: str | None = None,
    ):
        """Initialize a factor component.

        Args:
            var_name: Name of the categorical variable
            reference_level: Reference level for contrasts (not implemented yet)
            contrasts: Type of contrasts (only 'treatment' supported for now)
            name: Optional name for the term
        """
        self.var_name = var_name
        self.reference_level = reference_level
        self.contrasts = contrasts
        self._name = name

        # for now, only support treatment contrasts with no reference
        if reference_level is not None:
            raise NotImplementedError(
                "reference_level specification not yet implemented"
            )
        if contrasts != "treatment":
            raise NotImplementedError(
                f"contrasts='{contrasts}' not yet implemented; use 'treatment'"
            )

    @classmethod
    def from_formula(
        cls, func_name: str, args: list, kwargs: dict, term: str
    ) -> FactorComponent:
        if len(args) != 1:
            raise ValueError(
                f"{func_name}() expects exactly one argument, got {len(args)}: {args}"
            )

        var_name = args[0]
        reference_level = kwargs.pop("reference", None)
        contrasts = kwargs.pop("contrasts", "treatment")
        name = kwargs.pop("name", None)

        if kwargs:
            raise ValueError(
                f"{func_name}() does not accept additional arguments: {kwargs}"
            )

        return FactorComponent(
            var_name=var_name,
            reference_level=reference_level,
            contrasts=contrasts,
            name=name,
        )

    def formula_key(self, simplified: bool) -> str:
        if simplified:
            return f"factor({self.var_name})"
        else:
            parts = [self.var_name]
            if self.reference_level:
                parts.append(f"reference='{self.reference_level}'")
            if self.contrasts != "treatment":
                parts.append(f"contrasts='{self.contrasts}'")
            if self._name:
                parts.append(f"name='{self._name}'")
            return f"factor({', '.join(parts)})"

    def create_terms(
        self, registry: VariableRegistry, name_prefix: str = ""
    ) -> list[Term]:
        if self._name:
            term_name = self._name
        else:
            base_name = self.formula_key(simplified=True)
            term_name = f"{name_prefix}_{base_name}" if name_prefix else base_name

        # use registry's specialized method for dummy variables
        dummy_matrix_var = registry.get_dummy_vars(self.var_name, f"{term_name}_")

        return [
            LinearTerm(
                dummy_matrix_var,
                name=term_name,
                add_intercept=False,
                inference=gs.MCMCSpec(gs.IWLSKernel),
            )
        ]

    @property
    def includes_intercept(self) -> bool:
        return False

    @property
    def depends_on_vars(self) -> set[str]:
        return {self.var_name}


class MGCVComponent(FormulaComponent):
    """MGCV component for function calls using smoothcon."""

    def __init__(self, func_name: str, args: list, kwargs: dict, term: str):
        if len(args) != 1:
            raise ValueError(
                f"{func_name}() expects exactly one variable, got {len(args)}: {args}"
            )

        self.func_name = func_name
        self.variable_name = args[0]
        self._term = term

        # pop liesel-specific parameters (not for mgcv)
        kwargs_copy = kwargs.copy()
        self.term_name = kwargs_copy.pop("name", None)
        self.absorb_cons = kwargs_copy.pop("absorb_cons", True)
        self.ig_concentration = kwargs_copy.pop("ig_concentration", 0.01)
        self.ig_scale = kwargs_copy.pop("ig_scale", 0.01)

        # remaining kwargs are for mgcv
        self.mgcv_kwargs = kwargs_copy

    @classmethod
    def from_formula(
        cls, func_name: str, args: list, kwargs: dict, term: str
    ) -> MGCVComponent:
        """Create a MGCVComponent"""
        return MGCVComponent(func_name=func_name, args=args, kwargs=kwargs, term=term)

    def formula_key(self, simplified: bool) -> str:
        if simplified:
            return f"{self.func_name}({self.variable_name})"
        else:
            return self._term

    def create_terms(
        self, registry: VariableRegistry, name_prefix: str = ""
    ) -> list[Term]:
        import smoothcon

        if self.term_name:
            term_name = self.term_name  # ignore prefix for named components
        else:
            base_name = self.formula_key(simplified=True)
            term_name = f"{name_prefix}_{base_name}" if name_prefix else base_name

        # put strings in quotes for smoothcon
        smoothcon_kwargs = {
            k: f"'{v}'" if isinstance(v, str) else v
            for k, v in self.mgcv_kwargs.items()
        }

        # build smoothcon call string
        kwstr = ", ".join(f"{k}={v}" for k, v in smoothcon_kwargs.items())
        s_str = f"{self.func_name}({self.variable_name}"
        if kwstr:
            s_str += f", {kwstr}"
        s_str += ")"

        # use smoothcon to build basis and penalty matrices
        smooth = smoothcon.SmoothCon(
            s_str, data=registry.data, absorb_cons=self.absorb_cons
        )

        # get input variable
        input_var = registry.get_numeric_var(self.variable_name)

        # create basis using smoothcon
        basis = Basis(input_var, basis_fn=smooth, name=f"{term_name}_basis")

        # create smooth term with IWLS kernel
        smooth_term = SmoothTerm.new_ig(
            basis=basis,
            penalty=smooth.penalty,
            name=term_name,
            inference=gs.MCMCSpec(gs.IWLSKernel),
            ig_concentration=self.ig_concentration,
            ig_scale=self.ig_scale,
        )

        return [smooth_term]

    @property
    def includes_intercept(self) -> bool:
        return not self.absorb_cons

    @property
    def depends_on_vars(self) -> set[str]:
        return {self.variable_name}
