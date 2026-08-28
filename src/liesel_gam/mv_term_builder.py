from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self, cast

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pandas as pd

from .mv_predictor import (
    MultivariateContribution,
    MultivariateIntercept,
    MVAdditivePredictor,
)
from .mv_utils import (
    as_penalty_value,
    as_reparam_value,
    is_zero_penalty,
    scale_penalty_value,
)
from .registry import DictRegistry
from .term import (
    LinMixin,
    MultivariateStrctInteractionTerm,
    MultivariateStrctLinTerm,
    MultivariateStrctTerm,
    MultivariateTPTerm,
    StrctInteractionTerm,
    StrctLinTerm,
    StrctTerm,
)
from .term_builder import (
    InferenceTypes,
    TermBuilder,
    _biject_and_replace_star_gibbs_with,
    _format_name,
    _has_star_gibbs,
)
from .var import CatVar, ScaleIG, VarIGPrior

ArrayLike = jax.typing.ArrayLike
ScaleTypes = ScaleIG | lsl.Var | float | Literal["default"] | VarIGPrior


class MVTermBuilder:
    """Convenience builder for multivariate structured additive terms.

    The builder delegates covariate-basis construction to an ordinary
    :class:`.TermBuilder` and adds one shared cross-dimensional penalty and one
    term-specific cross-dimensional scale. Its term methods mirror
    :class:`.TermBuilder`; the ordinary ``scale`` controls smoothness within a
    dimension, while ``dimension_scale`` controls smoothness across dimensions.

    Usually, :meth:`from_predictor` is the most convenient constructor because it
    shares the predictor's penalty and any previously applied constraint.

    Parameters
    ----------
    registry
        Registry used to construct covariate bases.
    dimension_penalty
        Square, symmetric, positive-semidefinite cross-dimensional penalty.
    prefix_names_by
        Prefix applied to names created by the underlying term builder.
    default_inference
        Default inference specification for coefficient variables.
    default_scale_fn
        Default initializer for covariate-side scales.
    default_dimension_scale_fn
        Function that initializes a term-specific cross-dimensional scale, analogous
        to :class:`.TermBuilder`'s ``default_scale_fn``.
    default_scales_inference
        Default inference specification for covariate-side and cross-dimensional
        scale parameters.
    scale_penalty
        Whether to scale the cross-dimensional penalty to unit infinity norm.

    Examples
    --------
    >>> import liesel_gam as gam
    >>> data = gam.demo_data(20)
    >>> predictor = gam.MVAdditivePredictor.from_random_walk("delta", ndim=4)
    >>> predictor.constrain("sumzero_coef")
    MVAdditivePredictor(self.name='delta', 0 terms)
    >>> builder = gam.MVTermBuilder.from_predictor(
    ...     predictor, gam.TermBuilder.from_df(data)
    ... )
    >>> term = builder.ps("x_nonlin", k=7)
    >>> term.latent.value.shape, term.value.shape
    ((20, 3), (20, 4))
    """

    def __init__(
        self,
        registry: DictRegistry,
        dimension_penalty: ArrayLike | lsl.Value,
        prefix_names_by: str = "",
        default_inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel.untuned),
        default_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(1.0, 0.005),
        default_dimension_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(
            1.0, 0.005
        ),
        default_scales_inference: InferenceTypes | None = gs.MCMCSpec(gs.HMCKernel),
        scale_penalty: bool = True,
    ) -> None:
        marginal_builder = TermBuilder(
            registry=registry,
            prefix_names_by=prefix_names_by,
            default_inference=default_inference,
            default_scale_fn=default_scale_fn,
        )
        self._initialize(
            marginal_builder=marginal_builder,
            dimension_penalty=dimension_penalty,
            dimension_reparam=None,
            default_dimension_scale_fn=default_dimension_scale_fn,
            default_scales_inference=default_scales_inference,
            scale_penalty=scale_penalty,
            predictor=None,
        )

    def _initialize(
        self,
        *,
        marginal_builder: TermBuilder,
        dimension_penalty: ArrayLike | lsl.Value,
        dimension_reparam: ArrayLike | lsl.Value | None,
        default_dimension_scale_fn: Callable[[], lsl.Var] | VarIGPrior,
        default_scales_inference: InferenceTypes | None,
        scale_penalty: bool,
        predictor: MVAdditivePredictor | None,
    ) -> None:
        self.marginal_builder = marginal_builder
        self.registry = marginal_builder.registry
        self.names = marginal_builder.names
        self.bases = marginal_builder.bases
        self.default_inference = marginal_builder.default_inference
        self._default_dimension_scale_fn = default_dimension_scale_fn
        self.default_scales_inference = default_scales_inference
        self.predictor = predictor

        self._dimension_penalty = as_penalty_value(dimension_penalty)
        if scale_penalty:
            scale_penalty_value(self._dimension_penalty)
        self._dimension_reparam = as_reparam_value(
            dimension_reparam,
            latent_ndim=int(self._dimension_penalty.value.shape[-1]),
        )

    @classmethod
    def from_term_builder(
        cls,
        term_builder: TermBuilder,
        dimension_penalty: ArrayLike | lsl.Value,
        default_dimension_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(
            1.0, 0.005
        ),
        default_scales_inference: InferenceTypes | None = gs.MCMCSpec(gs.HMCKernel),
        scale_penalty: bool = True,
    ) -> Self:
        """Initialize from an existing ordinary term builder.

        Parameters
        ----------
        term_builder
            Builder that supplies the registry, naming state, and marginal terms.
        dimension_penalty
            Square cross-dimensional penalty.
        default_dimension_scale_fn
            Default initializer for term-specific cross-dimensional scales.
        default_scales_inference
            Default inference specification for scale parameters.
        scale_penalty
            Whether to scale ``dimension_penalty`` to unit infinity norm.

        Examples
        --------
        >>> marginal = TermBuilder.from_dict({"x": [0.0, 1.0]})
        >>> builder = MVTermBuilder.from_term_builder(marginal, jnp.eye(2))
        >>> builder.ndim
        2
        """
        obj = cls.__new__(cls)
        obj._initialize(
            marginal_builder=term_builder,
            dimension_penalty=dimension_penalty,
            dimension_reparam=None,
            default_dimension_scale_fn=default_dimension_scale_fn,
            default_scales_inference=default_scales_inference,
            scale_penalty=scale_penalty,
            predictor=None,
        )
        return obj

    @classmethod
    def from_predictor(
        cls,
        predictor: MVAdditivePredictor,
        term_builder: TermBuilder,
        default_dimension_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(
            1.0, 0.005
        ),
        default_scales_inference: InferenceTypes | None = gs.MCMCSpec(gs.HMCKernel),
    ) -> Self:
        """Initialize from a predictor and an ordinary term builder.

        Parameters
        ----------
        predictor
            Predictor whose penalty and reparameterization are shared by new terms.
        term_builder
            Builder that supplies the registry, naming state, and marginal terms.
        default_dimension_scale_fn
            Default initializer for term-specific cross-dimensional scales.
        default_scales_inference
            Default inference specification for scale parameters.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2)
        >>> marginal = TermBuilder.from_dict({"x": [0.0, 1.0]})
        >>> builder = MVTermBuilder.from_predictor(predictor, marginal)
        >>> builder.dimension_penalty is predictor.penalty
        True
        """
        obj = cls.__new__(cls)
        obj._initialize(
            marginal_builder=term_builder,
            dimension_penalty=predictor.penalty,
            dimension_reparam=predictor.dimension_reparam,
            default_dimension_scale_fn=default_dimension_scale_fn,
            default_scales_inference=default_scales_inference,
            scale_penalty=False,
            predictor=predictor,
        )
        return obj

    @classmethod
    def from_df(
        cls,
        data: pd.DataFrame,
        dimension_penalty: ArrayLike | lsl.Value,
        prefix_names_by: str = "",
        default_inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel.untuned),
        default_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(1.0, 0.005),
        default_dimension_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(
            1.0, 0.005
        ),
        default_scales_inference: InferenceTypes | None = gs.MCMCSpec(gs.HMCKernel),
        scale_penalty: bool = True,
    ) -> Self:
        """Initialize from a pandas DataFrame.

        Parameters
        ----------
        data
            DataFrame used by the underlying :class:`.TermBuilder`.
        dimension_penalty
            Square cross-dimensional penalty.
        prefix_names_by
            Prefix applied to names created by the builder.
        default_inference
            Default inference specification for coefficient variables.
        default_scale_fn
            Default initializer for covariate-side scales.
        default_dimension_scale_fn
            Default initializer for term-specific cross-dimensional scales.
        default_scales_inference
            Default inference specification for scale parameters.
        scale_penalty
            Whether to scale ``dimension_penalty`` to unit infinity norm.

        Examples
        --------
        >>> data = pd.DataFrame({"x": [0.0, 1.0]})
        >>> MVTermBuilder.from_df(data, jnp.eye(3)).ndim
        3
        """
        term_builder = TermBuilder.from_df(
            data,
            prefix_names_by=prefix_names_by,
            default_inference=default_inference,
            default_scale_fn=default_scale_fn,
        )
        return cls.from_term_builder(
            term_builder,
            dimension_penalty=dimension_penalty,
            default_dimension_scale_fn=default_dimension_scale_fn,
            default_scales_inference=default_scales_inference,
            scale_penalty=scale_penalty,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        dimension_penalty: ArrayLike | lsl.Value,
        **kwargs: Any,
    ) -> Self:
        """Initialize with a :class:`.DictRegistry` around ``data``.

        Construct :class:`.DictRegistry` or :class:`.PandasRegistry` directly for
        custom conversion or pandas missing-data handling. Nested mappings are not
        aligned; use :meth:`from_df` for a DataFrame or
        ``dataframe.to_dict("list")`` when converting one manually.

        Parameters
        ----------
        data
            Source values keyed by registry name.
        dimension_penalty
            Square cross-dimensional penalty.
        **kwargs
            Additional arguments forwarded to :class:`MVTermBuilder`.

        Examples
        --------
        >>> MVTermBuilder.from_dict({"x": [0.0, 1.0]}, jnp.eye(2)).ndim
        2
        """
        registry = DictRegistry(
            data,
            prefix_names_by=kwargs.get("prefix_names_by", ""),
        )
        return cls(registry, dimension_penalty, **kwargs)

    @property
    def dimension_penalty(self) -> lsl.Value:
        """The shared cross-dimensional penalty.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict({"x": [0.0, 1.0]}, jnp.eye(2))
        >>> builder.dimension_penalty.value.tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        """
        return self._dimension_penalty

    @property
    def dimension_reparam(self) -> lsl.Value:
        """Matrix mapping latent contributions to their full dimension.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict({"x": [0.0, 1.0]}, jnp.eye(2))
        >>> builder.dimension_reparam.value.tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        """
        return self._dimension_reparam

    @property
    def ndim(self) -> int:
        """Full output dimension.

        Examples
        --------
        >>> MVTermBuilder.from_dict({"x": [0.0, 1.0]}, jnp.eye(3)).ndim
        3
        """
        return int(self.dimension_reparam.value.shape[0])

    @property
    def latent_ndim(self) -> int:
        """Number of unconstrained cross-dimensional coordinates.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 3)
        >>> predictor.constrain("sumzero_coef")
        MVAdditivePredictor(self.name='delta', 0 terms)
        >>> builder = MVTermBuilder.from_predictor(
        ...     predictor, TermBuilder.from_dict({"x": [0.0, 1.0]})
        ... )
        >>> builder.latent_ndim
        2
        """
        return int(self.dimension_penalty.value.shape[-1])

    def labels_to_integers(self, newdata: dict[str, Any]) -> dict[str, Any]:
        """Encode categorical labels for prediction paths that require codes.

        Models containing :class:`.CatVar` accept labels directly in ``newdata``;
        this compatibility helper is not normally needed for those models.

        Parameters
        ----------
        newdata
            Replacement values keyed by source-value name.

        Examples
        --------
        >>> data = pd.DataFrame({"group": pd.Categorical(["a", "b", "a"])})
        >>> builder = MVTermBuilder.from_df(data, jnp.eye(2))
        >>> _ = builder.ri("group", scale=1.0, dimension_scale=1.0)
        >>> builder.labels_to_integers({"group": ["b", "a"]})["group"].tolist()
        [1, 0]
        """
        return self.marginal_builder.labels_to_integers(newdata)

    def _get_inference(
        self,
        inference: InferenceTypes | None | Literal["default"] = "default",
    ) -> InferenceTypes | None:
        return self.marginal_builder._get_inference(inference)

    def _get_scales_inference(
        self,
        inference: InferenceTypes | None | Literal["default"] = "default",
    ) -> InferenceTypes | None:
        if inference == "default":
            return self.default_scales_inference
        return inference

    def _finalize(self, term):
        if self.predictor is not None:
            self.predictor._lock_structure()
        return term

    def init_dimension_scale(
        self,
        scale: ScaleTypes,
        term_name: str,
    ) -> lsl.Var:
        """Initialize and name a scalar cross-dimensional scale.

        Parameters
        ----------
        scale
            Scale specification or ``"default"``.
        term_name
            Term name used to derive the scale name.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict({"x": [0.0, 1.0]}, jnp.eye(2))
        >>> float(builder.init_dimension_scale(2.0, "f(x)").value)
        2.0
        """
        if is_zero_penalty(self.dimension_penalty):
            if not (isinstance(scale, str) and scale == "default"):
                raise ValueError(
                    "dimension_scale is not identified when dimension_penalty is zero."
                )
            constant_name = self.names.create(
                _append_if_named(term_name, "_dimension_scale_constant"),
                apply_prefix=False,
            )
            return lsl.Var.new_value(jnp.asarray(1.0), name=constant_name)

        if isinstance(scale, str):
            if scale != "default":
                raise ValueError(f"Unknown dimension scale option: {scale!r}.")
            if isinstance(self._default_dimension_scale_fn, VarIGPrior):
                scale_spec: lsl.Var | ScaleIG = ScaleIG(
                    value=self._default_dimension_scale_fn.value,
                    concentration=self._default_dimension_scale_fn.concentration,
                    scale=self._default_dimension_scale_fn.scale,
                    name="{x}",
                    variance_name="{x}^2",
                )
            else:
                scale_spec = self._default_dimension_scale_fn()
        elif isinstance(scale, VarIGPrior):
            scale_spec = ScaleIG(
                value=scale.value,
                concentration=scale.concentration,
                scale=scale.scale,
                name="{x}",
                variance_name="{x}^2",
            )
        elif isinstance(scale, float):
            scale_spec = lsl.Var.new_value(jnp.asarray(scale), name="{x}")
        elif isinstance(scale, lsl.Var | ScaleIG):
            scale_spec = scale
        else:
            raise TypeError(f"Unexpected dimension scale type: {type(scale)}")

        if jnp.asarray(scale_spec.value).size != 1:
            raise ValueError(
                "A cross-dimensional scale must be scalar, "
                f"got size {jnp.asarray(scale_spec.value).size}."
            )

        if scale_spec.name:
            scale_name = self.names.psi(term_name)
            scale_spec = _format_name(scale_spec, fill=scale_name)
        return scale_spec

    def _prepare_scale(
        self,
        scale: lsl.Var | lsl.Node | None,
        scales_inference: InferenceTypes | None,
    ) -> None:
        if not isinstance(scale, lsl.Var):
            return
        if isinstance(scale, ScaleIG):
            _biject_and_replace_star_gibbs_with(
                scale,
                scales_inference,
                override_none_inference=True,
            )
        elif _has_star_gibbs(scale):
            _biject_and_replace_star_gibbs_with(scale, scales_inference)

    @staticmethod
    def _reject_factor_scale(kwargs: dict[str, Any]) -> None:
        if kwargs.get("factor_scale", False):
            raise NotImplementedError(
                "factor_scale=True is not implemented for multivariate terms."
            )

    def _wrap_marginal(
        self,
        marginal: StrctTerm,
        *,
        dimension_scale: ScaleTypes = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = "default",
        term_class: type[MultivariateStrctTerm] = MultivariateStrctTerm,
    ) -> MultivariateStrctTerm:
        scale = self.init_dimension_scale(dimension_scale, marginal.name)
        resolved_scales_inference = self._get_scales_inference(scales_inference)
        self._prepare_scale(marginal.scale, resolved_scales_inference)
        self._prepare_scale(scale, resolved_scales_inference)

        input_names = ",".join(
            StrctInteractionTerm._input_obs(StrctInteractionTerm._get_bases([marginal]))
        )
        basis_name = self.names.create(f"BMV({input_names})")
        coef_name = self.names.param("\\gamma", marginal.name)

        term = term_class(
            marginal,
            dimension_penalties=[self.dimension_penalty],
            dimension_scales=[scale],
            dimension_reparam=self.dimension_reparam,
            name=marginal.name,
            inference=self._get_inference(inference),
            coef_name=coef_name,
            basis_name=basis_name,
        )
        if is_zero_penalty(self.dimension_penalty):
            term.dimension_scale = None
        if isinstance(marginal, LinMixin) and isinstance(term, LinMixin):
            if marginal._model_spec is not None:
                term.model_spec = marginal.model_spec
            if marginal._mappings is not None:
                term.mappings = marginal.mappings
            term.column_names = marginal.column_names
        return self._finalize(term)

    def _call_and_wrap(
        self,
        method: str,
        *args,
        dimension_scale: ScaleTypes = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = "default",
        term_class: type[MultivariateStrctTerm] = MultivariateStrctTerm,
        **kwargs,
    ) -> MultivariateStrctTerm:
        self._reject_factor_scale(kwargs)
        marginal = getattr(self.marginal_builder, method)(
            *args,
            inference=inference,
            **kwargs,
        )
        return self._wrap_marginal(
            marginal,
            dimension_scale=dimension_scale,
            inference=inference,
            scales_inference=scales_inference,
            term_class=term_class,
        )

    def lin(
        self,
        *args: Any,
        prior: lsl.Dist | None = None,
        dimension_scale: ScaleTypes = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = "default",
        **kwargs: Any,
    ) -> MultivariateStrctLinTerm:
        """
        Initialize a multivariate linear term from a formula or :class:`.LinBasis`.

        When a ``LinBasis`` is supplied, this method attaches the required zero
        marginal penalty to that same object. The basis may therefore no longer be
        accepted by :meth:`.TermBuilder.lin` afterwards.

        Parameters
        ----------
        *args
            Formula or :class:`.LinBasis` and any positional arguments forwarded to
            :meth:`.TermBuilder.lin`.
        prior
            Scalar-term prior. Custom priors are not supported for multivariate
            linear terms.
        dimension_scale
            Cross-dimensional scale specification.
        inference
            Inference specification for the multivariate coefficient.
        scales_inference
            Inference specification for scale parameters.
        **kwargs
            Additional arguments forwarded to :meth:`.TermBuilder.lin`.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict({"x": [0.0, 1.0, 2.0]}, jnp.eye(2))
        >>> builder.lin("x", dimension_scale=1.0).value.shape
        (3, 2)
        """
        if prior is not None:
            raise NotImplementedError(
                "A custom scalar prior cannot be combined generically with the "
                "cross-dimensional penalty."
            )
        linear = getattr(self.marginal_builder, "lin")(
            *args, prior=None, inference=None, **kwargs
        )
        # This mutates a supplied LinBasis; revisit if multivariate bases need reuse.
        linear.basis.update_penalty(jnp.zeros((linear.nbases, linear.nbases)))
        proxy = StrctLinTerm(
            linear.basis,
            penalty=linear.basis.penalty,
            scale=lsl.Var.new_value(jnp.asarray(1.0)),
            name=linear.name,
            inference=None,
            coef_name="",
        )
        if linear._model_spec is not None:
            proxy.model_spec = linear.model_spec
        if linear._mappings is not None:
            proxy.mappings = linear.mappings
        proxy.column_names = linear.column_names
        return cast(
            MultivariateStrctLinTerm,
            self._wrap_marginal(
                proxy,
                dimension_scale=dimension_scale,
                inference=inference,
                scales_inference=scales_inference,
                term_class=MultivariateStrctLinTerm,
            ),
        )

    def slin(self, *args: Any, **kwargs: Any) -> MultivariateStrctLinTerm:
        """Build a multivariate structured linear term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.slin`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.slin` and multivariate
            wrapping, including ``dimension_scale`` and ``scales_inference``.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict({"x": [0.0, 1.0, 2.0]}, jnp.eye(2))
        >>> builder.slin("x", scale=1.0, dimension_scale=1.0).value.shape
        (3, 2)
        """
        return cast(
            MultivariateStrctLinTerm,
            self._call_and_wrap(
                "slin", *args, term_class=MultivariateStrctLinTerm, **kwargs
            ),
        )

    def cr(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate cubic regression spline.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.cr`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.cr` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {"x": jnp.linspace(0.0, 1.0, 12)}, jnp.eye(2)
        ... )
        >>> builder.cr("x", k=5, scale=1.0, dimension_scale=1.0).value.shape
        (12, 2)
        """
        return self._call_and_wrap("cr", *args, **kwargs)

    def cs(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate shrinkage cubic regression spline.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.cs`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.cs` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {"x": jnp.linspace(0.0, 1.0, 12)}, jnp.eye(2)
        ... )
        >>> builder.cs("x", k=5, scale=1.0, dimension_scale=1.0).value.shape
        (12, 2)
        """
        return self._call_and_wrap("cs", *args, **kwargs)

    def cc(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate cyclic cubic regression spline.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.cc`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.cc` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {"x": jnp.linspace(0.0, 1.0, 12)}, jnp.eye(2)
        ... )
        >>> builder.cc("x", k=5, scale=1.0, dimension_scale=1.0).value.shape
        (12, 2)
        """
        return self._call_and_wrap("cc", *args, **kwargs)

    def bs(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate B-spline term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.bs`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.bs` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {"x": jnp.linspace(0.0, 1.0, 12)}, jnp.eye(2)
        ... )
        >>> builder.bs("x", k=5, scale=1.0, dimension_scale=1.0).value.shape
        (12, 2)
        """
        return self._call_and_wrap("bs", *args, **kwargs)

    def ps(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate P-spline term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.ps`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.ps` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {"x": jnp.linspace(0.0, 1.0, 12)}, jnp.eye(2)
        ... )
        >>> builder.ps("x", k=5, scale=1.0, dimension_scale=1.0).value.shape
        (12, 2)
        """
        return self._call_and_wrap("ps", *args, **kwargs)

    def np(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate exclusively nonlinear P-spline term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.np`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.np` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {"x": jnp.linspace(0.0, 1.0, 12)}, jnp.eye(2)
        ... )
        >>> builder.np("x", k=5, scale=1.0, dimension_scale=1.0).value.shape
        (12, 2)
        """
        return self._call_and_wrap("np", *args, **kwargs)

    def cp(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate cyclic P-spline term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.cp`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.cp` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {"x": jnp.linspace(0.0, 1.0, 12)}, jnp.eye(2)
        ... )
        >>> builder.cp("x", k=5, scale=1.0, dimension_scale=1.0).value.shape
        (12, 2)
        """
        return self._call_and_wrap("cp", *args, **kwargs)

    def ri(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate random-intercept term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.ri`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.ri` and multivariate wrapping.

        Examples
        --------
        >>> data = pd.DataFrame({"group": pd.Categorical(["a", "b"] * 3)})
        >>> builder = MVTermBuilder.from_df(data, jnp.eye(2))
        >>> builder.ri("group", scale=1.0, dimension_scale=1.0).value.shape
        (6, 2)
        """
        return self._call_and_wrap("ri", *args, **kwargs)

    def mrf(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate Markov-random-field term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.mrf`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.mrf` and multivariate wrapping.

        Examples
        --------
        >>> data = pd.DataFrame({"region": pd.Categorical(["a", "b", "c"] * 2)})
        >>> builder = MVTermBuilder.from_df(data, jnp.eye(2))
        >>> penalty = jnp.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
        >>> term = builder.mrf(
        ...     "region",
        ...     penalty=penalty,
        ...     penalty_labels=["a", "b", "c"],
        ...     scale=1.0,
        ...     dimension_scale=1.0,
        ... )
        >>> term.value.shape
        (6, 2)
        """
        return self._call_and_wrap("mrf", *args, **kwargs)

    def f(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate term from a custom basis function.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.f`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.f` and multivariate wrapping.

        Examples
        --------
        >>> def linear_basis(values):
        ...     x = jnp.squeeze(values, axis=-1)
        ...     return jnp.column_stack((jnp.ones_like(x), x))
        >>> builder = MVTermBuilder.from_dict({"x": [0.0, 1.0, 2.0]}, jnp.eye(2))
        >>> term = builder.f(
        ...     "x",
        ...     basis_fn=linear_basis,
        ...     penalty=jnp.eye(2),
        ...     scale=1.0,
        ...     dimension_scale=1.0,
        ...     use_callback=False,
        ... )
        >>> term.value.shape
        (3, 2)
        """
        return self._call_and_wrap("f", *args, **kwargs)

    def kriging(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate kriging term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.kriging`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.kriging` and multivariate
            wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {
        ...         "x": jnp.linspace(0.0, 1.0, 12),
        ...         "z": jnp.linspace(1.0, 2.0, 12),
        ...     },
        ...     jnp.eye(2),
        ... )
        >>> term = builder.kriging("x", "z", k=5, scale=1.0, dimension_scale=1.0)
        >>> term.value.shape
        (12, 2)
        """
        return self._call_and_wrap("kriging", *args, **kwargs)

    def tp(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate thin-plate spline term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.tp`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.tp` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {
        ...         "x": jnp.linspace(0.0, 1.0, 12),
        ...         "z": jnp.linspace(1.0, 2.0, 12),
        ...     },
        ...     jnp.eye(2),
        ... )
        >>> term = builder.tp("x", "z", k=5, scale=1.0, dimension_scale=1.0)
        >>> term.value.shape
        (12, 2)
        """
        return self._call_and_wrap("tp", *args, **kwargs)

    def ts(self, *args: Any, **kwargs: Any) -> MultivariateStrctTerm:
        """Build a multivariate shrinkage thin-plate spline term.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`.TermBuilder.ts`.
        **kwargs
            Keyword arguments for :meth:`.TermBuilder.ts` and multivariate wrapping.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {
        ...         "x": jnp.linspace(0.0, 1.0, 12),
        ...         "z": jnp.linspace(1.0, 2.0, 12),
        ...     },
        ...     jnp.eye(2),
        ... )
        >>> term = builder.ts("x", "z", k=5, scale=1.0, dimension_scale=1.0)
        >>> term.value.shape
        (12, 2)
        """
        return self._call_and_wrap("ts", *args, **kwargs)

    def intercept(
        self,
        scale: ScaleTypes = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        prefix: str = "",
        name: str | None = None,
    ) -> MultivariateIntercept:
        """Build a basis-free multivariate intercept.

        Parameters
        ----------
        scale
            Cross-dimensional scale specification.
        inference
            Inference specification for the intercept coefficient.
        prefix
            Prefix added to the returned intercept name.
        name
            Optional explicit intercept name.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict({"x": [0.0, 1.0]}, jnp.eye(2))
        >>> builder.intercept(scale=1.0).value.shape
        (2,)
        """
        generated_name = self.names.create(prefix + (name or "intercept"))
        scale_var = (
            None
            if is_zero_penalty(self.dimension_penalty)
            else self.init_dimension_scale(scale, generated_name)
        )
        coef_name = self.names.param("\\gamma", generated_name)
        return self._finalize(
            MultivariateIntercept(
                dimension_penalty=self.dimension_penalty,
                dimension_reparam=self.dimension_reparam,
                scale=scale_var,
                name=generated_name,
                inference=self._get_inference(inference),
                coef_name=coef_name,
            )
        )

    def _extract_marginals(self, marginals: Sequence[lsl.Var]) -> list[StrctTerm]:
        if not marginals:
            raise ValueError("At least one tensor marginal is required.")
        extracted = []
        for marginal in marginals:
            if isinstance(marginal, MultivariateStrctTerm):
                if len(marginal.marginal_terms) != 1:
                    raise ValueError(
                        "A multivariate tensor marginal must wrap exactly one "
                        "ordinary marginal term."
                    )
                marginal_penalty = marginal.dimension_penalty
                if marginal_penalty is None:
                    raise ValueError(
                        "A tensor marginal must have one dimension penalty."
                    )
                if not jnp.allclose(
                    marginal_penalty.value, self.dimension_penalty.value
                ):
                    raise ValueError(
                        "Tensor marginal uses a different dimension penalty."
                    )
                if not jnp.allclose(
                    marginal.dimension_reparam.value, self.dimension_reparam.value
                ):
                    raise ValueError("Tensor marginal uses a different constraint.")
                extracted.append(marginal.marginal_terms[0])
            elif isinstance(marginal, StrctTerm):
                extracted.append(marginal)
            else:
                raise TypeError(f"Unsupported tensor marginal type: {type(marginal)}")

        for marginal in extracted:
            if marginal.scale_is_factored:
                raise NotImplementedError(
                    "Scale-factored marginals are not supported in multivariate terms."
                )
        return extracted

    def _init_common_scale(
        self,
        common_scale,
        term_name: str,
        scales_inference,
    ):
        if common_scale is None:
            return None
        scale = self.marginal_builder.init_scale(common_scale, term_name)
        self._prepare_scale(scale, self._get_scales_inference(scales_inference))
        return scale

    def tx(
        self,
        *marginals: lsl.Var,
        common_scale: ScaleTypes | None = None,
        dimension_scale: ScaleTypes = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = "default",
        prefix: str = "",
        name: str | None = None,
    ) -> MultivariateStrctInteractionTerm:
        """Build a factorized multivariate tensor interaction.

        Parameters
        ----------
        *marginals
            Ordinary or single-marginal multivariate structured terms.
        common_scale
            Optional shared covariate-side scale for all marginals.
        dimension_scale
            Cross-dimensional scale specification.
        inference
            Inference specification for the interaction coefficient.
        scales_inference
            Inference specification for scale parameters.
        prefix
            Prefix added to names created for the interaction.
        name
            Optional explicit term name.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {
        ...         "x": jnp.linspace(0.0, 1.0, 8),
        ...         "z": jnp.linspace(1.0, 2.0, 8),
        ...     },
        ...     jnp.eye(2),
        ... )
        >>> sx = builder.ps("x", k=5, scale=1.0, dimension_scale=1.0)
        >>> sz = builder.ps("z", k=5, scale=1.0, dimension_scale=1.0)
        >>> builder.tx(sx, sz, dimension_scale=1.0).value.shape
        (8, 2)
        """
        scalar_marginals = self._extract_marginals(marginals)
        input_names = ",".join(
            StrctInteractionTerm._input_obs(
                StrctInteractionTerm._get_bases(scalar_marginals)
            )
        )
        generated_name = self.names.create(prefix + f"tx({input_names})")
        term_name = prefix + name if name is not None else generated_name
        common_scale_var = self._init_common_scale(
            common_scale, term_name, scales_inference
        )
        if common_scale_var is not None:
            for marginal in scalar_marginals:
                marginal.replace_scale(common_scale_var)

        resolved_scales_inference = self._get_scales_inference(scales_inference)
        for marginal in scalar_marginals:
            self._prepare_scale(marginal.scale, resolved_scales_inference)
        dimension_scale_var = self.init_dimension_scale(dimension_scale, term_name)
        self._prepare_scale(dimension_scale_var, resolved_scales_inference)

        term = MultivariateStrctInteractionTerm(
            *scalar_marginals,
            dimension_penalties=[self.dimension_penalty],
            dimension_scales=[dimension_scale_var],
            dimension_reparam=self.dimension_reparam,
            name=term_name,
            inference=self._get_inference(inference),
            coef_name=self.names.param("\\gamma", term_name),
        )
        if is_zero_penalty(self.dimension_penalty):
            term.dimension_scale = None
        return self._finalize(term)

    def tf(
        self,
        *marginals: lsl.Var,
        common_scale: ScaleTypes | None = None,
        dimension_scale: ScaleTypes = "default",
        order: Sequence[int] | None = None,
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = "default",
        prefix: str = "",
        name: str | None = None,
        group_terms_by_order: bool = False,
    ) -> MultivariateTPTerm:
        """Build a full multivariate tensor-product term.

        Parameters
        ----------
        *marginals
            Ordinary or single-marginal multivariate structured terms.
        common_scale
            Optional shared covariate-side scale for all marginals.
        dimension_scale
            Cross-dimensional scale specification.
        order
            Interaction orders to include. By default all orders are included.
        inference
            Inference specification for coefficient variables.
        scales_inference
            Inference specification for scale parameters.
        prefix
            Prefix added to names created for the tensor product.
        name
            Optional explicit term name.
        group_terms_by_order
            Whether to expose an intermediate sum for each interaction order.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {
        ...         "x": jnp.linspace(0.0, 1.0, 8),
        ...         "z": jnp.linspace(1.0, 2.0, 8),
        ...     },
        ...     jnp.eye(2),
        ... )
        >>> sx = builder.ps("x", k=5, scale=1.0, dimension_scale=1.0)
        >>> sz = builder.ps("z", k=5, scale=1.0, dimension_scale=1.0)
        >>> builder.tf(sx, sz, dimension_scale=1.0).value.shape
        (8, 2)
        """
        scalar_marginals = self._extract_marginals(marginals)
        input_names = ",".join(
            StrctInteractionTerm._input_obs(
                StrctInteractionTerm._get_bases(scalar_marginals)
            )
        )
        generated_name = self.names.create(prefix + f"tf({input_names})")
        term_name = prefix + name if name is not None else generated_name
        common_scale_var = self._init_common_scale(
            common_scale, term_name, scales_inference
        )

        resolved_scales_inference = self._get_scales_inference(scales_inference)
        for marginal in scalar_marginals:
            self._prepare_scale(marginal.scale, resolved_scales_inference)
        dimension_scale_var = self.init_dimension_scale(dimension_scale, term_name)
        self._prepare_scale(dimension_scale_var, resolved_scales_inference)

        term = MultivariateTPTerm(
            *scalar_marginals,
            common_scale=common_scale_var,
            dimension_penalties=[self.dimension_penalty],
            dimension_scales=[dimension_scale_var],
            dimension_reparam=self.dimension_reparam,
            order=order,
            inference=self._get_inference(inference),
            names_prefix=prefix,
            tx_name="tx",
            tf_name="tf",
            coef_name=r"\gamma",
            group_terms_by_order=group_terms_by_order,
        )
        term.name = term_name
        term.latent.name = self.names.create(_append_if_named(term_name, "_latent"))
        term.latent.value_node.name = self.names.create(
            term.latent.name + "_value_node"
        )
        term.latent.var_value_node.name = self.names.create(
            term.latent.name + "_var_value_node"
        )

        for order_, subterms in term.terms_by_order.items():
            for subterm in subterms:
                subterm.name = self.names.create(prefix + f"tx({subterm.xnames})")
                subterm.coef.name = self.names.param("\\gamma", subterm.name)
                if order_ == 1:
                    subterm.basis.name = self.names.create(f"BMV({subterm.xnames})")
                    subterm.basis.value_node.name = self.names.create(
                        subterm.basis.name + "_value_node"
                    )
                    subterm.basis.var_value_node.name = self.names.create(
                        subterm.basis.name + "_var_value_node"
                    )
                subterm.latent.name = self.names.create(
                    _append_if_named(subterm.name, "_latent")
                )
                subterm.latent.value_node.name = self.names.create(
                    subterm.latent.name + "_value_node"
                )
                subterm.latent.var_value_node.name = self.names.create(
                    subterm.latent.name + "_var_value_node"
                )

        if group_terms_by_order:
            for group in term.term_groups.values():
                group.name = self.names.create(group.name)
                group.value_node.name = self.names.create(
                    group.name + "_value_node", apply_prefix=False
                )
                group.var_value_node.name = self.names.create(
                    group.name + "_var_value_node", apply_prefix=False
                )

        if is_zero_penalty(self.dimension_penalty):
            term.dimension_scale = None
        return self._finalize(term)

    def rs(
        self,
        x: str | lsl.Var | StrctTerm,
        cluster: str | CatVar,
        *,
        dimension_scale: ScaleTypes = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = "default",
        prefix: str = "",
        name: str | None = None,
        **kwargs: Any,
    ) -> MultivariateContribution:
        """Build a multivariate random-slope effect.

        Parameters
        ----------
        x
            Named numeric source value or variable multiplied by the cluster effect.
        cluster
            Registry name or named :class:`.CatVar` identifying clusters.
        dimension_scale
            Cross-dimensional scale specification.
        scales_inference
            Inference specification for scale parameters.
        prefix
            Prefix added to the returned effect name.
        name
            Optional explicit effect name.
        **kwargs
            Additional arguments forwarded to :meth:`ri`.

        Examples
        --------
        >>> data = pd.DataFrame(
        ...     {
        ...         "x": jnp.arange(6.0),
        ...         "group": pd.Categorical(["a", "b"] * 3),
        ...     }
        ... )
        >>> builder = MVTermBuilder.from_df(data, jnp.eye(2))
        >>> builder.rs("x", "group", scale=1.0, dimension_scale=1.0).value.shape
        (6, 2)
        """
        self._reject_factor_scale(kwargs)
        if isinstance(x, str):
            x_var = self.registry.get_numeric_obs(x)
            x_name = x
            x_input = x_var
        elif isinstance(x, CatVar):
            raise TypeError("Random-slope 'x' must be numeric, not a CatVar.")
        elif isinstance(x, lsl.Var):
            x_name = x.name
            x_input = getattr(x, "latent", x)
        else:
            raise TypeError(f"Unsupported random-slope input: {type(x)}")

        if not x_name:
            raise ValueError("A variable supplied as random-slope 'x' must be named.")
        x_value = jnp.asarray(x_input.value)
        if not jnp.issubdtype(x_value.dtype, jnp.number):
            raise TypeError("Random-slope 'x' must be numeric.")
        if x_value.ndim not in (1, 2) or x_value.size == 0:
            raise ValueError(
                "Multivariate random-slope 'x' must be nonempty and 1D or 2D."
            )

        random_intercept = self.ri(
            cluster,
            dimension_scale=dimension_scale,
            scales_inference=scales_inference,
            **kwargs,
        )
        cluster_shape = random_intercept.latent.value.shape
        if x_value.shape[0] != cluster_shape[0]:
            raise ValueError(
                "Random-slope 'x' and 'cluster' must have the same length."
            )
        if x_value.ndim == 2 and x_value.shape[1:] != cluster_shape[1:]:
            raise ValueError(
                "A matrix-valued random-slope 'x' must match the latent cluster shape."
            )

        def multiply(x, cluster):
            if x.ndim == cluster.ndim - 1:
                x = jnp.expand_dims(x, -1)
            return x * cluster

        cluster_name = cluster if isinstance(cluster, str) else cluster.name
        generated_name = self.names.create(prefix + f"rs({x_name}|{cluster_name})")
        term_name = prefix + name if name is not None else generated_name
        latent = lsl.Var.new_calc(
            multiply,
            x=x_input,
            cluster=random_intercept.latent,
            name=_append_if_named(term_name, "_latent"),
        )
        effect = MultivariateContribution(
            latent,
            dimension_reparam=self.dimension_reparam,
            dimension_penalty=self.dimension_penalty,
            name=term_name,
        )
        setattr(effect, "random_intercept", random_intercept)
        return self._finalize(effect)

    def vc(
        self,
        x: str | lsl.Var,
        by: lsl.Var,
        *,
        dimension_scale: ScaleTypes = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = "default",
        prefix: str = "",
        name: str | None = None,
    ) -> MultivariateContribution:
        """Build a multivariate varying-coefficient effect.

        Parameters
        ----------
        x
            Named numeric source value or variable multiplying ``by``.
        by
            Ordinary or multivariate structured term that supplies the varying
            coefficient.
        dimension_scale
            Cross-dimensional scale used when wrapping an ordinary ``by`` term.
        inference
            Inference specification used when wrapping an ordinary ``by`` term.
        scales_inference
            Inference specification for scale parameters.
        prefix
            Prefix added to the returned effect name.
        name
            Optional explicit effect name.

        Examples
        --------
        >>> builder = MVTermBuilder.from_dict(
        ...     {
        ...         "x": jnp.linspace(0.0, 1.0, 8),
        ...         "z": jnp.linspace(1.0, 2.0, 8),
        ...     },
        ...     jnp.eye(2),
        ... )
        >>> by = builder.ps("z", k=5, scale=1.0, dimension_scale=1.0)
        >>> builder.vc("x", by).value.shape
        (8, 2)
        """
        if isinstance(by, MultivariateStrctTerm):
            if by.dimension_penalty is None or not jnp.allclose(
                by.dimension_penalty.value, self.dimension_penalty.value
            ):
                raise ValueError(
                    "Varying-coefficient term uses a different dimension penalty."
                )
            if not jnp.allclose(
                by.dimension_reparam.value, self.dimension_reparam.value
            ):
                raise ValueError(
                    "Varying-coefficient term uses a different constraint."
                )
            by_mv = by
        elif isinstance(by, StrctTerm):
            by_mv = self._wrap_marginal(
                by,
                dimension_scale=dimension_scale,
                inference=inference,
                scales_inference=scales_inference,
            )
        else:
            raise TypeError(f"Unsupported varying-coefficient term: {type(by)}")

        if isinstance(x, CatVar):
            raise TypeError("Varying-coefficient 'x' must be numeric, not a CatVar.")
        x_var = self.bases._get_var_and_value(x)[0]

        def multiply(x, by):
            return jnp.expand_dims(x, -1) * by

        generated_name = self.names.create(prefix + x_var.name + "*" + by_mv.name)
        term_name = prefix + name if name is not None else generated_name
        latent = lsl.Var.new_calc(
            multiply,
            x=x_var,
            by=by_mv.latent,
            name=_append_if_named(term_name, "_latent"),
        )
        effect = MultivariateContribution(
            latent,
            dimension_reparam=self.dimension_reparam,
            dimension_penalty=self.dimension_penalty,
            name=term_name,
        )
        setattr(effect, "by", by_mv)
        return self._finalize(effect)


def _append_if_named(name: str, suffix: str) -> str:
    return name + suffix if name else ""
