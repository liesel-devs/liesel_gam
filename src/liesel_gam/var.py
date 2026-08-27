from __future__ import annotations

import copy
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.core import Tracer

from .category_mapping import CategoryMapping
from .kernel import init_star_ig_gibbs, init_star_ig_gibbs_factored

InferenceTypes = Any
Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class VarIGPrior(NamedTuple):
    concentration: float
    scale: float
    value: float = 1.0


def _append_name(name: str, append: str) -> str:
    if name == "":
        return ""
    else:
        return name + append


def _ensure_var_or_node(
    x: lsl.Var | lsl.Node | ArrayLike,
    name: str | None,
) -> lsl.Var | lsl.Node:
    """
    If x is an array, creates a new observed variable.
    """
    if isinstance(x, lsl.Var | lsl.Node):
        x_var = x
    else:
        name = name if name is not None else ""
        x_var = lsl.Var.new_obs(jnp.asarray(x), name=name)

    if name is not None and x_var.name != name:
        raise ValueError(f"{x_var.name=} and {name=} are incompatible.")

    return x_var


def _ensure_value(
    x: lsl.Var | lsl.Node | ArrayLike,
    name: str | None,
) -> lsl.Var | lsl.Node:
    """
    If x is an array, creates a new value node.
    """
    if isinstance(x, lsl.Var | lsl.Node):
        x_var = x
    else:
        name = name if name is not None else ""
        x_var = lsl.Value(jnp.asarray(x), _name=name)

    if name is not None and x_var.name != name:
        raise ValueError(f"{x_var.name=} and {name=} are incompatible.")

    return x_var


class UserVar(lsl.Var):
    """
    A :class:`liesel.model.Var`, adapted for subclassing.

    What differentiates this from the basic :class:`liesel.model.Var` is just that
    the alternative constructors

    - :meth:`liesel.model.Var.new_obs`
    - :meth:`liesel.model.Var.new_param`
    - :meth:`liesel.model.Var.new_calc`
    - :meth:`liesel.model.Var.new_value`

    are disabled to avoid potential errors when variables are subclassed and intended
    to be initialized directly.
    """

    @classmethod
    def new_calc(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_obs(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_param(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_value(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )


class CatVar(UserVar):
    """An observed variable that stores categorical labels as integer codes.

    Parameters
    ----------
    labels
        A nonempty rectangular array of hashable, nonmissing, non-integer labels.
        Pandas Series, pandas categoricals, and NumPy arrays are supported. All
        entries share one category mapping.
    name
        Optional variable name. Builders require a directly supplied ``CatVar`` to
        be named and one-dimensional.
    categories
        Optional ordered non-integer categories. They may include unobserved
        categories. Without this argument, categories are inferred in sorted order;
        pandas categorical order is preserved.
    unknown_category
        Optional non-integer catch-all label for unknown nonmissing labels. It is
        appended to the category mapping if necessary. The default ``None`` rejects
        unknown labels.
    dist
        Optional Liesel distribution for the encoded integer values.

    The ordinary constructor rejects integer labels because later integer inputs would
    be ambiguous with encoded category codes. Convert semantic integer labels to a
    non-integer representation first. Use :meth:`from_codes` only when the supplied
    integers are already contiguous, zero-based codes.

    A catch-all category does not accept invalid integer codes or missing values. It
    only handles unknown semantic labels.

    Label-valued replacement, prediction, and sampling inputs are converted on the
    host before compiled model operations. The JIT-compiled numerical path receives
    integer codes and therefore remains compatible with JAX transformations.

    .. note::

        Concrete code arrays are validated against the mapping. Traced integer arrays
        must already contain valid codes; dynamic bounds checks are not performed
        inside JAX transformations.

    Examples
    --------
    Strings, pandas Series, and NumPy string arrays can be used directly:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import liesel_gam as gam
    >>> gam.CatVar(["b", "a", "b"], name="group").value
    Array([1, 0, 1], dtype=int32)
    >>> gam.CatVar(pd.Series(["a", "b"]), name="group")
    CatVar(name="group")
    >>> gam.CatVar(np.array(["a", "b"]), name="group")
    CatVar(name="group")

    Explicit categories preserve their order and may include unused levels:

    >>> group = gam.CatVar(
    ...     ["control", "treated"],
    ...     categories=["control", "treated", "unused"],
    ...     name="group",
    ... )
    >>> dict(group.mapping.labels_to_codes_map)
    {'control': 0, 'treated': 1, 'unused': 2}

    Unknown labels can be mapped to an explicitly enabled catch-all category:

    >>> group = gam.CatVar(
    ...     ["control", "treated"],
    ...     unknown_category="other",
    ...     name="group",
    ... )
    >>> group.mapping.labels_to_codes(["new-level"]).tolist()
    [2]

    When used for a random intercept, all unknown labels share this one catch-all
    coefficient. If no training observation maps to it, that coefficient is informed
    by its prior and shared hyperparameters rather than directly by data.

    Use :meth:`from_codes` for existing encoded data. An optional distribution sees
    these codes:

    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> mapping = gam.CategoryMapping({"a": 0, "b": 1})
    >>> dist = lsl.Dist(tfd.Categorical, logits=jnp.zeros(2))
    >>> group = gam.CatVar.from_codes([0, 1], mapping=mapping, name="group", dist=dist)

    Models containing a ``CatVar`` accept labels in prediction data:

    >>> coef = lsl.Var.new_param(jnp.zeros(2), name="coef")
    >>> effect = lsl.Var.new_calc(
    ...     lambda group, coef: coef[group], group, coef, name="effect"
    ... )
    >>> model = lsl.Model(effect)
    >>> model.predict(
    ...     {"coef": jnp.array([[10.0, 20.0]])},
    ...     predict=["effect"],
    ...     newdata={"group": ["b", "a"]},
    ... )["effect"]
    Array([[20., 10.]], dtype=float32)

    Integer-valued labels are rejected with guidance:

    >>> gam.CatVar([20, 10])
    Traceback (most recent call last):
        ...
    TypeError: CatVar labels must not be integers; use strings or CatVar.from_codes().
    """

    def __init__(
        self,
        labels: Any,
        *,
        name: str = "",
        categories: Any = None,
        unknown_category: Any | None = None,
        dist: lsl.Dist | None = None,
    ) -> None:
        supplied_labels = [labels]
        if categories is not None:
            supplied_labels.append(categories)
        if unknown_category is not None:
            supplied_labels.append([unknown_category])
        for supplied in supplied_labels:
            labels_flat = np.asarray(supplied, dtype=object).reshape(-1)
            if any(
                isinstance(label, (int, np.integer))
                and not isinstance(label, (bool, np.bool_))
                for label in labels_flat
            ):
                raise TypeError(
                    "CatVar labels must not be integers; use strings or "
                    "CatVar.from_codes()."
                )

        self._mapping = CategoryMapping.from_labels(
            labels, categories, unknown_category=unknown_category
        )
        self._accepts_host_codes = False
        codes = jnp.asarray(self._mapping.labels_to_codes(labels))
        super().__init__(codes, dist=dist, name=name, convert=self._convert_value)
        self.observed = True

    @classmethod
    def from_codes(
        cls,
        codes: Any,
        *,
        mapping: CategoryMapping,
        name: str = "",
        dist: lsl.Dist | None = None,
    ) -> CatVar:
        """Create a categorical variable from already encoded integer codes.

        The mapping defines every valid contiguous, zero-based code and may include
        categories that are not observed in ``codes``. Later integer inputs are
        interpreted as codes, while non-integer inputs are converted as labels.

        Parameters
        ----------
        codes
            A nonempty rectangular array of integer category codes.
        mapping
            The category mapping that defines every valid code.
        name
            Optional variable name.
        dist
            Optional Liesel distribution for the encoded integer values.

        Examples
        --------
        >>> mapping = CategoryMapping({"a": 0, "b": 1})
        >>> group = CatVar.from_codes([1, 0], mapping=mapping, name="group")
        >>> group.value
        Array([1, 0], dtype=int32)
        """
        if not isinstance(mapping, CategoryMapping):
            raise TypeError("mapping must be a CategoryMapping.")

        codes_array = np.asarray(codes)
        if codes_array.size == 0:
            raise ValueError("Categorical codes must be nonempty.")
        if not np.issubdtype(codes_array.dtype, np.integer):
            raise TypeError("Categorical codes must have an integer dtype.")

        var = cls.__new__(cls)
        var._mapping = mapping
        var._accepts_host_codes = True
        encoded = jnp.asarray(mapping.to_codes(codes_array))
        lsl.Var.__init__(var, encoded, dist=dist, name=name, convert=var._convert_value)
        var.observed = True
        return var

    @property
    def mapping(self) -> CategoryMapping:
        """The category mapping shared by all entries."""
        return self._mapping

    def _convert_value(self, value: Any) -> jax.Array:
        if isinstance(value, Tracer):
            value = jnp.asarray(value)
            if value.size == 0:
                raise ValueError("Categorical values must be nonempty.")
            if not jnp.issubdtype(value.dtype, jnp.integer):
                raise TypeError("Traced CatVar values must be integer codes.")
            return value

        value_array = np.asarray(value)
        if value_array.size == 0:
            raise ValueError("Categorical values must be nonempty.")
        normalized_value = jax.tree.map(
            lambda item: (
                item.item() if isinstance(item, jax.Array) and item.ndim == 0 else item
            ),
            value,
        )

        is_integer_array = np.issubdtype(value_array.dtype, np.integer)
        if is_integer_array:
            is_internal_jax_value = isinstance(value, jax.Array)
            if not self._accepts_host_codes and not is_internal_jax_value:
                raise ValueError(
                    "Integer values are ambiguous for a label-mode CatVar; supply "
                    "non-integer labels or construct the variable with "
                    "CatVar.from_codes()."
                )
            return jnp.asarray(self.mapping.to_codes(normalized_value))
        return jnp.asarray(self.mapping.labels_to_codes(normalized_value))


class ScaleIG(UserVar):
    r"""
    A variable with an Inverse Gamma prior on its square.

    The variance parameter (i.e. the squared scale) is flagged as a parameter.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        Often called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        Often called ``b``.
    name
        Name of the variable.

    Notes
    -----

    This class assumes that this variable represents the scale parameter
    :math:`\tau` in a structured additive term prior as described in
    :class:`.StrctTerm`.

    This class allows for easy setup of Gibbs sampling for :math:`\tau^2` via
    :meth:`.setup_gibbs_inference`. The Gibbs sampler is defined as follows.

    We have

    .. math::

        \tau^2 \sim \operatorname{InverseGamma}(a, b),

    where a is the init argument ``concentration`` and b is the init argument
    ``scale`` for :class:`.ScaleIG`. The value of this variable (ScaleIG) is
    :math:`\tau = \sqrt{\tau^2}`.

    In a structured additive term,
    the coefficient :math:`\boldsymbol{\beta} \in \mathbb{R}^J`
    receives a potentially rank-deficient multivariate normal prior

    .. math::

        p(\boldsymbol{\beta}) \propto \left(\frac{1}{\tau^2}\right)^{
        \operatorname{rk}(\mathbf{K})/2}
        \exp \left(
        - \frac{1}{\tau^2} \boldsymbol{\beta}^\top \mathbf{K} \boldsymbol{\beta}
        \right).

    The full conditional distribution for :math:`\tau^2` is then an inverse Gamma
    distribtion:

    .. math::

        \tau^2 | \cdot \sim \operatorname{InverseGamma}(\tilde{a}, \tilde{b})

    with parameters

    .. math::

        \tilde{a}  & = a + 0.5 \operatorname{rk}(\mathbf{K}) \\
        \tilde{b}  & = b + 0.5 \boldsymbol{\beta}^\top \mathbf{K} \boldsymbol{\beta}.

    The Gibbs sampler for :math:`\tau^2` repeatedly draws from this full conditional.

    References
    -----------

    Section 9.6.3 in

    Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013). Regression—Models, methods
    and applications. Springer. https://doi.org/10.1007/978-3-642-34333-9

    """

    def __init__(
        self,
        value: float | Array,
        concentration: float | lsl.Var | lsl.Node | ArrayLike,
        scale: float | lsl.Var | lsl.Node | ArrayLike,
        name: str = "",
        variance_name: str = "",
    ):
        value = jnp.asarray(value)

        concentration_node = _ensure_value(
            concentration, name=_append_name(name, "_concentration")
        )
        scale_node = _ensure_value(scale, name=_append_name(name, "_scale"))

        prior = lsl.Dist(
            tfd.InverseGamma, concentration=concentration_node, scale=scale_node
        )

        variance_name = variance_name or _append_name(name, "_square")

        self._variance_param = lsl.Var.new_param(value**2, prior, name=variance_name)
        super().__init__(lsl.Calc(jnp.sqrt, self._variance_param), name=name)

    def setup_gibbs_inference(
        self, coef: lsl.Var, penalty: jax.typing.ArrayLike | None = None
    ) -> ScaleIG:
        r"""
        Sets up a :class:`liesel.goose.GibbsKernel` for this variable, assuming
        that it is used as the variance parameter in a structured additive term.

        See the docs for the class :class:`.ScaleIG` for a description of the
        Gibbs sampler.

        .. note::
            Usually, this method does not have to be called manually, when you are
            working
            with :class:`.StrctTernm` objects or initializing terms using
            :class:`.TermBuilder`.

        Parameters
        ----------
        coef
            Coefficient variable.
        penalty
            Penalty matrix. If ``None``, the penalty is assumed to be the identity
            matrix of a dimension fitting the coefficient dimension.

        See Also
        --------
        .StrctTerm : Structured additive term class.

        """
        if self.value.size != 1:
            raise ValueError(
                f"Gibbs sampler assumes scalar value, got size {self.value.size}."
            )
        init_gibbs = copy.copy(init_star_ig_gibbs)
        init_gibbs.__name__ = "StarVarianceGibbs"

        self._variance_param.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={"coef": coef, "scale": self, "penalty": penalty},
        )
        return self

    def setup_gibbs_inference_factored(
        self,
        scaled_coef: lsl.Var,
        latent_coef: lsl.Var,
        penalty: jax.typing.ArrayLike | None = None,
    ) -> ScaleIG:
        if self.value.size != 1:
            raise ValueError(
                f"Gibbs sampler assumes scalar value, got size {self.value.size}."
            )
        init_gibbs = copy.copy(init_star_ig_gibbs_factored)
        init_gibbs.__name__ = "StarVarianceGibbs"

        self._variance_param.inference = gs.MCMCSpec(
            init_star_ig_gibbs_factored,
            kernel_kwargs={
                "scaled_coef": scaled_coef,
                "latent_coef": latent_coef,
                "scale": self,
                "penalty": penalty,
            },
        )
        return self
