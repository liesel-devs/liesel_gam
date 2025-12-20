from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pandas as pd
import tensorflow_probability.substrates.jax.bijectors as tfb
from liesel.model.model import TemporaryModel

from .basis_builder import BasisBuilder
from .names import NameManager
from .registry import CategoryMapping, PandasRegistry
from .term import LinTerm, MRFTerm, RITerm, StrctLinTerm, StrctTensorProdTerm, StrctTerm
from .var import ScaleIG, VarIGPrior

InferenceTypes = Any

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

BasisTypes = Literal["tp", "ts", "cr", "cs", "cc", "bs", "ps", "cp", "gp"]


logger = logging.getLogger(__name__)


def labels_to_integers(newdata: dict, mappings: dict[str, CategoryMapping]) -> dict:
    # replace categorical inputs with their index representation
    # create combined input matrices from individual variables, if desired
    newdata = newdata.copy()

    # replace categorical variables by their integer representations
    for name, mapping in mappings.items():
        if name in newdata:
            newdata[name] = mapping.labels_to_integers(newdata[name])

    return newdata


class TermBuilder:
    r"""
    Initializes structured additive model terms.

    The terms returned by the methods of this class are all instances of
    :class:`liesel.model.Var`, or of its subclasses.

    Among other things, the term builder automatically assigns unique names to the
    created variables.

    Parameters
    ----------
    registry
        Provides an interface to a data frame used to set up the model terms.
    prefix_names_by
        Names created by this TermBuilder will be prefixed by the string supplied here.
    default_inference
        Defines the default inference specification for terms created by this builder.
        Note that this inference is only used for the coefficient variables
        of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for the
        scale variables (:attr:`.StrctTerm.scale`).
    default_scale_fn
        A function or :class:`.VarIGPrior` object that defines the default scale
        for structured additive terms initialized by this builder. If this is a
        function, it must take no arguments and return a :class:`liesel.model.Var`
        that acts as the scale. If it is a :class:`.VarIGPrior`, the default scale
        will be ``scale = sqrt(var)``, where
        ``var ~ InverseGamma(concentration, scale)``, with concentration and scale
        given by the :class:`.VarIGPrior` object. For most terms, this
        will mean that a fitting Gibbs sampler can be automatically set up for
        ``var``. The exceptions to this rule are :meth:`.ta`, :meth:`.tf`, and
        :meth:`.tx`. Note that, if you supply a custom default scale function, you
        should make sure that the ``inference`` attribute of your custom scale
        is defined, otherwise your custom scale may not be included in MCMC
        sampling.

    See Also
    --------

    .BasisBuilder : Initializes :class:`.Basis` objects with penalty matrices.

    Notes
    ------

    The terms created by this builder generally have the form

    .. math::
        s(\mathbf{x}_i) = \sum_{j=1}^J B_j(\mathbf{x}_i) \beta_j
        = \mathbf{b}(\mathbf{x}_i)^\top \boldsymbol{\beta}

    where

    - :math:`i=1, \dots, N` is the observation index,
    - :math:`\mathbf{x}_i^\top = [x_{i,1}, \dots, x_{i,M}]` are covariate
      observations, where :math:`M` denotes the number of covariates,
    - :math:`\mathbf{b}(\mathbf{x}_i)^\top = [B_1(\mathbf{x}_i),
      \dots, B_J(\mathbf{x}_i)]`
      are a set of basis function evaluations, and
    - :math:`\boldsymbol{\beta}^\top = [\beta_1, \dots, \beta_J]`
      are the corresponding coefficients.

    In many cases, :math:`\mathbf{x}_i` will consist
    of only one covariate, except for linear effects (:meth:`.lin`, :meth:`.slin`) or
    tensor product smooths (:meth:`.tf`, :meth:`.tx`).

    The basis matrix for such a term is

    .. math::

        \mathbf{B} = \begin{bmatrix}
        \mathbf{b}(\mathbf{x}_1)^\top \\
        \vdots \\
        \mathbf{b}(\mathbf{x}_N)^\top
        \end{bmatrix}.

    The coefficient receives a potentially rank-deficient multivariate normal prior

    .. math::

        p(\boldsymbol{\beta}) \propto \left(
        \frac{1}{\tau^2}\right)^{\operatorname{rk}(\mathbf{K})/2}
        \exp \left(
        - \frac{1}{\tau^2} \boldsymbol{\beta}^\top \mathbf{K} \boldsymbol{\beta}
        \right)

    with the potentially rank-deficient penalty matrix :math:`\mathbf{K}` of rank
    :math:`\operatorname{rk}(\mathbf{K})`. The variance
    parameter :math:`\tau^2` acts as an inverse smoothing parameter.

    The choice of basis functions :math:`B_j` and penalty matrix :math:`\mathbf{K}`
    determines the nature of the term.

    .. rubric:: Sampling specification

    By default, the coefficients for each term created with this termbuilder will
    be equipped with the TermBuilder's default inference specification.
    This default specification corresponds to a blockwise sampling scheme, with
    each term's coefficients forming a single block. See the examples for how to
    modify the default blockwise setup.


    .. rubric:: Overview of commonly used terms

    .. note::
        **Basic terms**

        - :meth:`.lin` : Linear term.
        - :meth:`.slin` : Linear term with iid penalty (ridge prior).
        - :meth:`.ps` : P-spline.
        - :meth:`.tp` : Thin plate spline.
        - :meth:`.ri` : Random intercept.
        - :meth:`.mrf` : Markov random field (discrete spatial effect).
        - :meth:`.kriging` : Low-rank gaussian process with fixed range.

        **Combined terms and tensor products**

        - :meth:`.rs` : Random slope.
        - :meth:`.vc` : Varying coefficient.
        - :meth:`.tx` : Tensor product interaction without main effects.
        - :meth:`.tf` : Full tensor product with main effects.

        **Specialized smooths**

        - :meth:`.np` : P-splines without linear trend.
        - :meth:`.cp` : Cyclic P-splines

        **Custom smooths**

        - :meth:`.f` : Supply your own basis function and penalty matrix.
        - :class:`.StrctTerm` : Initialize a term independetly, potentially supplying
          a constant basis matrix and your own penalty matrix.

    .. tip::

        If your model somewhere contains a categorical variable, pay attention
        to the method :meth:`labels_to_integers`; this helps you bring a ``newdata``
        dictionary into a form understood by :meth:`liesel.model.Model.predict`
        easily by turning string labels into their integer representations.

    Examples
    --------

    Basic example using defaults, initializing a P-spline:

    >>> import liesel_gam as gam
    >>> df = gam.demo_data(100)
    >>> tb = gam.TermBuilder.from_df(df)
    >>> tb.ps("x_nonlin", k=20)
    StrctTerm(name="ps(x_nonlin)")

    Changing the default inference. This still means, each term is sampled in a
    separate block.

    >>> import liesel.goose as gs
    >>> import liesel_gam as gam
    >>> df = gam.demo_data(100)
    >>> tb = gam.TermBuilder.from_df(df, default_inference=gs.MCMCSpec(gs.NUTSKernel))
    >>> tb.ps("x_nonlin", k=20)
    StrctTerm(name="ps(x_nonlin)")

    Changing default inference, such that all terms' coefficients are collected in the
    same block. Note that the scales are still sampled individually; by default with
    Gibbs kernels.

    >>> import liesel.goose as gs
    >>> import liesel_gam as gam
    >>> df = gam.demo_data(100)
    >>> tb = gam.TermBuilder.from_df(
    ...     df,
    ...     default_inference=gs.MCMCSpec(gs.NUTSKernel, kernel_group="1"),
    ... )
    >>> tb.ps("x_nonlin", k=20)
    StrctTerm(name="ps(x_nonlin)")

    Changing the default scale. Here, we change the default scales to have a HalfNormal
    prior, and sample them on inverse-softplus level using independent IWLS kernels.

    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.bijectors as tfb
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> import liesel_gam as gam

    >>> def scale_fn():
    ...     prior = lsl.Dist(
    ...         tfd.HalfNormal,
    ...         scale=jnp.array(20.0),
    ...     )
    ...     scale = lsl.Var.new_param(
    ...         jnp.array(0.1),
    ...         distribution=prior,
    ...         name="{x}",  # placeholder for the automatically generated name
    ...     )
    ...     scale.transform(
    ...         tfb.Softplus(),
    ...         inference=gs.MCMCSpec(gs.IWLSKernel.untuned),  # inference for scales
    ...         name="h({x})",  # {x} is a placeholder
    ...     )
    ...     return scale


    >>> df = gam.demo_data(100)
    >>> tb = gam.TermBuilder.from_df(df, default_scale_fn=scale_fn)
    >>> tb.ps("x_nonlin", k=20)
    StrctTerm(name="ps(x_nonlin)")

    Using a name prefix:

    >>> import liesel_gam as gam
    >>> df = gam.demo_data(100)
    >>> tb = gam.TermBuilder.from_df(df, prefix_names_by="loc.")
    >>> tb.ps("x_nonlin", k=20)
    StrctTerm(name="loc.ps(loc.x_nonlin)")

    If you don't want the name prefix to appear on the covariate names, too, initialize
    the :class:`.PandasRegistry` individually. This way, you can for example use the
    same registry for two TermBuilder instances.

    >>> import liesel_gam as gam
    >>> df = gam.demo_data(100)
    >>> registry = gam.PandasRegistry(df)

    >>> tb1 = gam.TermBuilder(registry, prefix_names_by="loc.")
    >>> tb1.ps("x_nonlin", k=20)
    StrctTerm(name="loc.ps(x_nonlin)")

    >>> tb2 = gam.TermBuilder(registry, prefix_names_by="scale.")
    >>> tb2.ps("x_nonlin", k=20)
    StrctTerm(name="scale.ps(x_nonlin)")

    """

    def __init__(
        self,
        registry: PandasRegistry,
        prefix_names_by: str = "",
        default_inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel.untuned),
        default_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(1.0, 0.005),
    ) -> None:
        self.registry = registry
        self.names = NameManager(prefix=prefix_names_by)
        self.bases = BasisBuilder(registry, names=self.names)
        self.default_inference = default_inference
        self._default_scale_fn = default_scale_fn

    def _get_inference(
        self,
        inference: InferenceTypes | None | Literal["default"] = "default",
    ) -> InferenceTypes | None:
        if inference == "default":
            return self.default_inference
        else:
            return inference

    def init_scale(
        self,
        scale: lsl.Var | ScaleIG | float | Literal["default"] | VarIGPrior,
        term_name: str,
    ) -> lsl.Var:
        """
        Initializes a scale variable with a term-related name.

        Parameters
        ----------
        scale
            Scale object.
        term_name
            Name of the term this scale corresponds to. If you supply a
            :class:`liesel.model.Var`, you can use the place- holder ``{x}`` in its name
            to allow this method to fill in the ``term_name``.

        Notes
        -----
        The behavior depends on the type of the ``scale`` argument.

        - If it is ``"default"``, the return will be created based on the
            ``default_scale_fn`` argument supplied to the TermBuilder upon
            initialization.
        - If it is a :class:`.VarIGPrior`, the return
            will be ``scale = sqrt(var)``, where ``var ~ InverseGamma(concentration,
            scale)``, with concentration and scale given by the :class:`.VarIGPrior`
            object. For most terms, this will mean that a fitting Gibbs sampler can be
            automatically set up for ``var``. The exceptions to this rule are
            :meth:`.ta`, :meth:`.tf`, and :meth:`.tx`.
        - If it is a ``float``, the return will be ``lsl.Var.new_value`` holding this
            float.
        - If it is a :class:`liesel.model.Var` object, the return will be this
            object. If you supply a :class:`liesel.model.Var`, you can use the place-
            holder ``{x}`` in its name to allow this method to fill in an automatically
            generated name based on the ``term_name``.
        """
        if scale == "default":
            if isinstance(self._default_scale_fn, VarIGPrior):
                scale_var: lsl.Var | ScaleIG = ScaleIG(
                    value=self._default_scale_fn.value,
                    concentration=self._default_scale_fn.concentration,
                    scale=self._default_scale_fn.scale,
                    name="{x}",
                    variance_name="{x}^2",
                )
            else:
                scale_var = self._default_scale_fn()
        elif isinstance(scale, VarIGPrior):
            scale_var = ScaleIG(
                value=scale.value,
                concentration=scale.concentration,
                scale=scale.scale,
                name="{x}",
                variance_name="{x}^2",
            )
        elif isinstance(scale, float):
            scale_var = lsl.Var.new_value(scale)
        elif isinstance(scale, lsl.Var | ScaleIG):
            scale_var = scale
        else:
            raise TypeError(f"Unexpected scale type: {type(scale)}")

        scale_name = self.names.tau(term_name)
        scale_var = _format_name(scale_var, fill=scale_name)
        return scale_var

    @classmethod
    def from_dict(
        cls,
        data: dict[str, ArrayLike],
        prefix_names_by: str = "",
        default_inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel.untuned),
        default_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(1.0, 0.005),
    ) -> TermBuilder:
        """
        Initializes a TermBuilder from a dictionary that holds the data.

        Internally, this will create a :class:`.PandasRegistry` with
        ``na_action="drop"``.

        The other arguments are passed on to the init.
        """
        return cls.from_df(
            pd.DataFrame(data),
            prefix_names_by=prefix_names_by,
            default_inference=default_inference,
            default_scale_fn=default_scale_fn,
        )

    @classmethod
    def from_df(
        cls,
        data: pd.DataFrame,
        prefix_names_by: str = "",
        default_inference: InferenceTypes | None = gs.MCMCSpec(gs.IWLSKernel.untuned),
        default_scale_fn: Callable[[], lsl.Var] | VarIGPrior = VarIGPrior(1.0, 0.005),
    ) -> TermBuilder:
        """
        Initializes a TermBuilder from a pandas dataframe.

        Internally, this will create a :class:`.PandasRegistry` with
        ``na_action="drop"``.

        The other arguments are passed on to the init.
        """
        registry = PandasRegistry(
            data, na_action="drop", prefix_names_by=prefix_names_by
        )
        return cls(
            registry,
            prefix_names_by=prefix_names_by,
            default_inference=default_inference,
            default_scale_fn=default_scale_fn,
        )

    def labels_to_integers(self, newdata: dict) -> dict:
        """
        Processes a ``newdata`` dictionary, replacing labels of caterogical variables
        with their integer representation, such that they can be understood by
        :meth:`liesel.model.Model.predict`.
        """
        return labels_to_integers(newdata, self.bases.mappings)

    # formula
    def lin(
        self,
        formula: str,
        prior: lsl.Dist | None = None,
        inference: InferenceTypes | None | Literal["default"] = "default",
        context: dict[str, Any] | None = None,
    ) -> LinTerm:
        """
        Linear term.

        Parameters
        ----------
        formula
            Right-hand side of a model formula, as understood by formulaic_. Most of
            formulaic's grammar_ is supported. See notes for details.
        prior
            An optional prior for this term's coefficient. The default is a constant
            prior.
        inference
            An optional :class:`liesel.goose.MCMCSpec` instance (or other valid
            inference object). If omitted, the term's default inference specification
            is used.
        context
            Dictionary of additional Python objects that should be made available to
            formulaic when constructing the design matrix. Gets passed to
            ``formulaic.ModelSpec.get_model_matrix()``.

        See Also
        --------
        .slin : Linear term with identity penalty matrix, leading to a ridge prior.

        Notes
        -----

        This term evaluates to :math:`\\mathbf{X}\\boldsymbol{\\beta}`, where
        :math:`\\mathbf{X}` is a linear-effect design matrix. The coefficient
        vector receives a constant prior by default,
        :math:`\\boldsymbol{\\beta} \\sim \\text{const}`, but a custom prior can be
        passed in the argument ``prior`` as a :class:`liesel.model.Dist`.

        The following formulaic syntax is supported:

        - ``+`` for adding a term
        - ``a:b`` for simple interactions
        - ``a*b`` for expanding to ``a + b + a:b``
        - ``(a + b)**n`` for n-th order interactions
        - ``a / b`` for nesting
        - ``C(a, ...)`` for categorical effects
        - ``b %in% a`` for inverted nesting
        - ``{a+1}`` for quoted Python code to be executed
        - ```weird name``` backtick-strings for weird names
        - Other transformations like ``center(a)``, ``scale(a)``, or ``lag(a)``, see
          grammar_.
        - Python functions

        Not supported:

        - String literals
        - Numeric literals
        - Wildcard ``"."``
        - ``\\|`` for splitting a formula
        - ``"~"`` in formula, since this method supports only the right-hand side of a
          Wilkinson formula.
        - ``1 +``, ``0 +``, or ``-1`` in formula, since intercept addition is handled
          via the argument ``include_intercept``.

        References
        ----------

        - Python library formulaic: https://matthewwardrop.github.io/formulaic/latest/

        Examples
        --------

        Simple example:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.lin("x_lin + x_nonlin + x_cat")
        LinBasis(name="X")

        Customized categorical encoding:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.lin("x_lin + x_nonlin + C(x_cat, contr.sum)")
        LinBasis(name="X")

        Interaction:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.lin("x_lin * x_cat")
        LinBasis(name="X")


        .. _formulaic: https://matthewwardrop.github.io/formulaic/latest/
        .. _grammar: https://matthewwardrop.github.io/formulaic/latest/guides/grammar/
        """

        include_intercept = False

        basis = self.bases.lin(
            formula,
            xname="",
            basis_name="X",
            include_intercept=include_intercept,
            context=context,
        )

        term_name = self.names.create("lin" + "(" + basis.name + ")")

        coef_name = self.names.beta(term_name)

        term = LinTerm(
            basis,
            prior=prior,
            name=term_name,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )

        term.model_spec = basis.model_spec
        term.mappings = basis.mappings
        term.column_names = basis.column_names

        return term

    def slin(
        self,
        formula: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        context: dict[str, Any] | None = None,
        factor_scale: bool = False,
    ) -> StrctLinTerm:
        """
        Linear term with an identity penalty matrix, leading to a ridge prior.

        Parameters
        ----------
        formula
            Right-hand side of a model formula, as understood by formulaic_. Most of
            formulaic's grammar_ is supported. See notes for details.
        prior
            An optional prior for this term's coefficient. The default is a constant
            prior.
        inference
            An optional :class:`liesel.goose.MCMCSpec` instance (or other valid
            inference object). If omitted, the term's default inference specification
            is used.
        context
            Dictionary of additional Python objects that should be made available to
            formulaic when constructing the design matrix. Gets passed to
            ``formulaic.ModelSpec.get_model_matrix()``.

        See Also
        --------
        .lin : Linear term with constant prior.

        Notes
        -----

        The following formulaic syntax is supported:

        - ``+`` for adding a term
        - ``a:b`` for simple interactions
        - ``a*b`` for expanding to ``a + b + a:b``
        - ``(a + b)**n`` for n-th order interactions
        - ``a / b`` for nesting
        - ``C(a, ...)`` for categorical effects
        - ``b %in% a`` for inverted nesting
        - ``{a+1}`` for quoted Python code to be executed
        - ```weird name``` backtick-strings for weird names
        - Other transformations like ``center(a)``, ``scale(a)``, or ``lag(a)``, see
          grammar_.
        - Python functions

        Not supported:

        - String literals
        - Numeric literals
        - Wildcard ``"."``
        - ``\\|`` for splitting a formula
        - ``"~"`` in formula, since this method supports only the right-hand side of a
          Wilkinson formula.
        - ``1 +``, ``0 +``, or ``-1`` in formula, since intercept addition is handled
          via the argument ``include_intercept``.

        References
        ----------

        - Python library formulaic: https://matthewwardrop.github.io/formulaic/latest/

        Examples
        --------

        Simple example:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.slin("x_lin")
        StrctLinTerm(name="slin(X)")


        .. _formulaic: https://matthewwardrop.github.io/formulaic/latest/
        .. _grammar: https://matthewwardrop.github.io/formulaic/latest/guides/grammar/
        """
        include_intercept = False

        basis = self.bases.lin(
            formula,
            xname="",
            basis_name="X",
            include_intercept=include_intercept,
            context=context,
        )
        basis._penalty = lsl.Value(jnp.eye(basis.nbases))

        fname = self.names.create("slin" + "(" + basis.name + ")")

        term = StrctLinTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=self.names.beta(fname),
        )
        if factor_scale:
            term.factor_scale()

        term.model_spec = basis.model_spec
        term.mappings = basis.mappings
        term.column_names = basis.column_names

        return term

    def cr(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        Cubic regression spline.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .cs : Cubic regression splines with additinal shrinkage on the null space.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.


        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.cr("x_nonlin", k=20)
        StrctTerm(name="cr(x_nonlin)")
        """
        basis = self.bases.cr(
            x=x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cr", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def cs(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        Cubic regression spline with additional null space shrinkage.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .cr : Cubic regression splines.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.


        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.cs("x_nonlin", k=20)
        StrctTerm(name="cs(x_nonlin)")
        """
        basis = self.bases.cs(
            x=x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cs", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def cc(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        Cyclic version of cubic regression spline.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .cr : Cubic regression splines.
        .cs : Cubic regression splines with additinal shrinkage on the null space.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.


        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.cc("x_nonlin", k=20)
        StrctTerm(name="cc(x_nonlin)")
        """
        basis = self.bases.cc(
            x=x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cc", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def bs(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        basis_degree: int = 3,
        penalty_order: int | Sequence[int] = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        B-spline with integrated squared derivative penalties.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        basis_degree
            Degree of the polynomials used in the B-spline basis function. Default is 3
            for cubic B-splines.
        penalty_order
            Order of the penalty. If this is a sequence of integers, a
            penalty of the integer's order is added for each entry in the sequence.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .ps : P-splines.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis
        and penalty. The mgcv documentation provides further details.

        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.bs("x_nonlin", k=20)
        StrctTerm(name="bs(x_nonlin)")
        """
        basis = self.bases.bs(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("bs", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    # P-spline
    def ps(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        P-spline: A B-spline with a discrete penalty matrix.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        basis_degree
            Degree of the polynomials used in the B-spline basis function. Default is 3
            for cubic B-splines.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .np : P-spline without linear trend.
        .cp : Cyclic P-spline.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.

        References
        ----------
        - Lang, S., & Brezger, A. (2004). Bayesian P-splines. Journal of Computational
          and Graphical Statistics, 13(1), 183–212.
          https://doi.org/10.1198/1061860043010
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.ps("x_nonlin", k=20)
        StrctTerm(name="ps(x_nonlin)")

        The default is a constrained basis:

        >>> tb.ps("x_nonlin", k=20).basis.value.shape
        (100, 19)

        The constraint can be turned off by passing ``absorb_cons=False``:

        >>> tb.ps("x_nonlin", k=20, absorb_cons=False).basis.value.shape
        (100, 20)
        """
        basis = self.bases.ps(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("ps", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def np(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        P-spline without linear trend.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        basis_degree
            Degree of the polynomials used in the B-spline basis function. Default is 3
            for cubic B-splines.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .ps : P-spline.
        .cp : Cyclic P-spline.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.

        References
        ----------
        - Lang, S., & Brezger, A. (2004). Bayesian P-splines. Journal of Computational
          and Graphical Statistics, 13(1), 183–212.
          https://doi.org/10.1198/1061860043010
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.np("x_nonlin", k=20)
        StrctTerm(name="np(x_nonlin)")
        """
        basis = self.bases.ps(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=False,
            diagonal_penalty=False,
            scale_penalty=False,
            basis_name="B",
        )

        basis.constrain("constant_and_linear")
        if scale_penalty:
            basis.scale_penalty()
        if diagonal_penalty:
            basis.diagonalize_penalty()

        fname = self.names.fname("np", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def cp(
        self,
        x: str,
        *,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        Cyclic P-spline.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        basis_degree
            Degree of the polynomials used in the B-spline basis function. Default is 3
            for cubic B-splines.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .ps : P-spline.
        .np : P-spline without linear trend.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.

        References
        ----------
        - Lang, S., & Brezger, A. (2004). Bayesian P-splines. Journal of Computational
          and Graphical Statistics, 13(1), 183–212.
          https://doi.org/10.1198/1061860043010
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.cp("x_nonlin", k=20)
        StrctTerm(name="cp(x_nonlin)")
        """
        basis = self.bases.cp(
            x=x,
            k=k,
            basis_degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("cp", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis=basis,
            penalty=basis.penalty,
            scale=self.init_scale(scale, fname),
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    # random intercept
    def ri(
        self,
        cluster: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        penalty: ArrayLike | None = None,
        factor_scale: bool = False,
    ) -> RITerm:
        r"""
        Random intercept.

        Parameters
        ----------
        cluster
            Name of the cluster variable.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        penalty
            Custom penalty matrix to use. Default is an iid penalty.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .rs : Random slope.
        .BasisBuilder : Initializes the basis and penalty.

        Notes
        ------
        If the penalty is iid, then each column of the basis consists only of binary
        (0/1) entries, and each row has only one non-zero entry. In this case it is not
        necessary to store the full matrix in memory and evaluate the term as a dot
        product ``basis @ coef``.

        Instead, we can simply store a 1d array of indices, identifying the nonzero
        column for each row of the basis matrix, and use this index to access the
        corresponding coefficient. This scenario is common for independent random
        intercepts.

        This method returns such a sparse representation of the random intercept
        basis if ``penalty=None``.

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.ri("x_cat")
        RITerm(name="ri(x_cat)")

        """
        basis = self.bases.ri(cluster=cluster, basis_name="B", penalty=penalty)

        fname = self.names.fname("ri", basis.x.name)
        coef_name = self.names.beta(fname)

        term = RITerm(
            basis=basis,
            penalty=basis.penalty,
            coef_name=coef_name,
            inference=self._get_inference(inference),
            scale=self.init_scale(scale, fname),
            name=fname,
        )

        if factor_scale:
            term.factor_scale()

        mapping = self.bases.mappings[cluster]
        term.mapping = mapping
        term.labels = list(mapping.labels_to_integers_map)
        nparams = len(mapping.labels_to_integers_map)

        if basis.penalty is None and nparams != basis.nbases:
            # this takes care of increasing the parameter number in case this term
            # covers unobserved clusters
            term.coef.value = jnp.zeros(nparams)

        return term

    # random scaling
    def rs(
        self,
        x: str | StrctTerm | LinTerm,
        cluster: str,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        penalty: ArrayLike | None = None,
        factor_scale: bool = False,
    ) -> lsl.Var:
        r"""
        Random slope.

        Create a Liesel variable that evaluates to ``x * ri(cluster)``, where
        ``ri(cluster)`` is a
        random intercept initialized via :meth:`.ri` and ``x`` is either a covariate
        directly, or a smooth term.

        The ``scale`` argument of this method is the random intercept's
        scale, it is passed to :meth:`.ri`. The same goes for ``penalty`` and
        ``factor_scale``.

        Parameters
        ----------
        x
            Name of input variable, or a smooth represented by a :class:`.StrctTerm`.
        cluster
            Name of the cluster variable.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        penalty
            Custom penalty matrix to use. Default is an iid penalty.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .ri : Random intercept.
        .BasisBuilder : Initializes the basis and penalty.


        Examples
        ---------

        Random slope:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.rs(x="x_lin", cluster="x_cat")
        Var(name="rs(x_lin|x_cat)")

        Random scaling of a smooth term:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> psx = tb.ps("x_nonlin", k=20)
        >>> tb.rs(x=psx, cluster="x_cat")
        Var(name="rs(ps(x_nonlin)|x_cat)")

        """

        ri = self.ri(
            cluster=cluster,
            scale=scale,
            inference=self._get_inference(inference),
            penalty=penalty,
            factor_scale=factor_scale,
        )

        if isinstance(x, str):
            x_var = self.registry.get_numeric_obs(x)
            xname = x
        else:
            x_var = x
            xname = x_var.name

        fname = self.names.create("rs(" + xname + "|" + cluster + ")")
        term = lsl.Var.new_calc(
            lambda x, cluster: x * cluster,
            x=x_var,
            cluster=ri,
            name=fname,
        )
        return term

    # varying coefficient
    def vc(
        self,
        x: str,
        by: StrctTerm,
    ) -> lsl.Var:
        r"""
        Varying coefficient term.

        Parameters
        ----------
        x
            Name of input variable.
        by
            Smooth term, a :class:`.StrctTerm` that represents the smoothly varying
            coefficient of this term, for example a P-spline :meth:`.ps`.

        Notes
        -----

        A varying coefficient term can be written as

        .. math::

            x \beta(z),

        where :math:`x` is a covariate, and :math:`\beta(z)` is the linear effect of
        this covariate, which smoothly varies as a function of another variable
        :math:`z`.

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> psx = tb.ps("x_nonlin", k=20)
        >>> tb.vc(x="x_lin", by=psx)
        Var(name="x_lin*ps(x_nonlin)")
        """
        fname = self.names.create(x + "*" + by.name)
        x_var = self.registry.get_obs(x)

        term = lsl.Var.new_calc(
            lambda x, by: x * by,
            x=x_var,
            by=by,
            name=fname,
        )
        return term

    # general smooth with MGCV bases
    def _s(
        self,
        *x: str,
        k: int,
        bs: BasisTypes,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        m: str = "NA",
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        basis = self.bases._s(
            *x,
            k=k,
            bs=bs,
            m=m,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname(bs, basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis,
            penalty=basis.penalty,
            name=fname,
            coef_name=coef_name,
            scale=self.init_scale(scale, fname),
            inference=self._get_inference(inference),
        )
        if factor_scale:
            term.factor_scale()
        return term

    # markov random field
    def mrf(
        self,
        x: str,
        k: int = -1,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        polys: dict[str, ArrayLike] | None = None,
        nb: Mapping[str, ArrayLike | list[str] | list[int]] | None = None,
        penalty: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> MRFTerm:
        r"""
        Gaussian Markov random field.

        The preferred way to initialize these is by supplying ``polys``, because this
        enables plotting via :func:`.plot_regions`.

        Parameters
        ----------
        x
            Name of the region variable.
        k
            If ``-1``, this is a "full-rank" (up to identifiability constraint) Markov
            random field. If ``k`` is an integer smaller than the number of unique
            regions, a low-rank field will be returned, see Wood (2017), Sections 5.8.1
            and 5.4.2.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        polys
            Dictionary of arrays. The keys of the dict are the region labels. The
            corresponding values define the region by defining polygons. The
            neighborhood structure can be inferred from this polygon information.
        nb
            Dictionary of array. The keys of the dict are the region labels. The
            corresponding values indicate the neighbors of the region. If the values are
            lists or arrays of strings, the values are the labels of the neighbors. If
            they are lists or arrays of integers, the values are the indices of the
            neighbors. Indices correspond to regions based on an alphabetical ordering
            of regions.
        penalty
            If a penalty is supplied explicitly, it takes precedence over a potential
            penalty derived from both nb and polys.
        penalty_labels
            If a penalty is supplied explicitly, labels must also be specified. The
            labels create the association between penalty columns and region labels. The
            values of this sequence should be the string labels of unique regions in
            ``x``.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------
        .plot_regions : Plots MCMC results on a map of the regions.
        .plot_polys : Plots a map based on polygons.
        .plot_forest : Plots regions with uncertainty in a forest plot.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

        Returns
        -------

            Comments on the additional attributes available on the returned
            :class:`.MRFTerm` variable:

            - If either polys or nb are supplied, the returned term will contain
              information in :attr:`.MRFTerm.neighbors`.
            - If only a penalty matrix is supplied, the returned MRFSpec will *not*
              contain information in :attr:`.MRFTerm.neighbors`.
            - :attr:`.MRFTerm.mapping` contains the map of region labels to integer
              codes.
            - :attr:`.MRFTerm.labels` contains the region labels.
            - Returning the label order only makes sense if the basis is *not*
              reparameterized, because only then we have a clear correspondence of
              parameters to labels. If the basis is reparameterized, with
              ``absorb_cons=True`` or of low rank with ``k ≠ -1``, there is no such
              correspondence in a clear way, so the label order
              in :attr:`.MRFTerm.ordered_labels` is None.


        Examples
        --------
        >>> import liesel_gam as gam

        >>> df = gam.demo_data(n=100)
        >>> nb = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        >>> print(df.x_cat.unique())
        ['a' 'b' 'c']

        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.mrf("x_cat", nb=nb)
        MRFTerm(name="mrf(x_cat)")


        References
        ----------
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        """
        basis = self.bases.mrf(
            x=x,
            k=k,
            polys=polys,
            nb=nb,
            penalty=penalty,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("mrf", basis.x.name)
        coef_name = self.names.beta(fname)
        term = MRFTerm(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=self.init_scale(scale, fname),
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()

        term.polygons = basis.mrf_spec.polys
        term.neighbors = basis.mrf_spec.nb
        if basis.mrf_spec.ordered_labels is not None:
            term.ordered_labels = basis.mrf_spec.ordered_labels

        term.labels = list(basis.mrf_spec.mapping.labels_to_integers_map)
        term.mapping = basis.mrf_spec.mapping

        return term

    # general basis function + penalty smooth
    def f(
        self,
        *x: str,
        basis_fn: Callable[[Array], Array],
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: ArrayLike | None = None,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        General :class:`.StrctTerm`, initialized by passing a custom basis function.

        .. note::
            You can use :meth:`.StrctTerm.constrain` to apply linear constraints to your
            custom term.

        Parameters
        ----------
        *x
            Variable number of input variable names. Can be one or more.
        basis_fn
            Basis function. Must take a 2d-array as input and return a 2d array.

        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        use_callback
            If *True*, the basis function is evaluated using a Python callback,
            which means that it does not have to be jit-compatible via JAX. This also
            means that the basis must remain constant throughout estimation.
            Passed on to :class:`.Basis`.
        cache_basis
            If ``True`` the computed basis is cached in a persistent
            calculation node (``lsl.Calc``), which avoids re-computation
            when not required. Passed on to :class:`.Basis`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .StrctTerm.constrain : Apply constraints to a term after initialization.
        .StrctTerm.diagonalize_penalty : Diagonalize the penalty of a term after
          initialization.
        .StrctTerm.scale_penalty : Scale the penalty of a term after
          initialization.
        .BasisBuilder.basis : Used by this method to set up the basis.

        Examples
        --------
        Manually set up a P-spline:

        >>> from liesel.contrib.splines import (
        ...     basis_matrix,
        ...     equidistant_knots,
        ...     pspline_penalty,
        ... )
        >>> import liesel_gam as gam

        >>> df = gam.demo_data(n=100)
        >>> tb = gam.TermBuilder.from_df(df)

        Set up basis function:

        >>> knots = equidistant_knots(df["x_nonlin"].to_numpy(), n_param=20)
        >>> pen = pspline_penalty(d=20)
        >>> def bspline_basis(x_mat):
        ...     # x_mat is shape (n, 1)
        ...     x_vec = x_mat.squeeze()  # shape (n,)
        ...     return basis_matrix(x_vec, knots=knots)

        Initialize the term:

        >>> fx = tb.f("x_nonlin", basis_fn=bspline_basis, penalty=pen)

        The term currently has a basis of dimension (100, 20), since it is
        unconstrained.

        >>> fx.basis.value.shape
        (100, 20)

        You can use :meth:`.StrctTerm.constrain` to apply a constraint:

        >>> fx.constrain("sumzero_term")
        StrctTerm(name="f(x_nonlin)")

        Now the basis dimension is reduced:

        >>> fx.basis.value.shape
        (100, 19)

        """
        basis = self.bases.basis(
            *x,
            basis_fn=basis_fn,
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=penalty,
            basis_name="B",
        )

        fname = self.names.fname("f", basis.x.name)
        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=self.init_scale(scale, fname),
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def kriging(
        self,
        *x: str,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        kernel_name: Literal[
            "spherical",
            "power_exponential",
            "matern1.5",
            "matern2.5",
            "matern3.5",
        ] = "matern1.5",
        linear_trend: bool = True,
        range: float | None = None,
        power_exponential_power: float = 1.0,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        Gaussian process models with a fixed range parameter in a
        basis-penalty-parameterization, often referred to as Kriging.

        Parameters
        ----------
        *x
            Name of input variables (one or more).
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        kernel_name
            Selects the kernel / covariance function to use.
        linear_trend
            Whether to include or remove a linear trend.
        range
            Range parameter. If ``None``, estimated as in Kamman & Wand (2003).
        power_exponential_power
            Power for the power exponential kernel.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.

        See Also
        --------

        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

        References
        ----------
        - Kammann, E. E. and M.P. Wand (2003) Geoadditive Models. Applied Statistics
          52(1):1-18.
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.kriging("x_nonlin", k=20)
        StrctTerm(name="kriging(x_nonlin)")

        """
        basis = self.bases.kriging(
            *x,
            k=k,
            kernel_name=kernel_name,
            linear_trend=linear_trend,
            range=range,
            power_exponential_power=power_exponential_power,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("kriging", basis.x.name)
        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=self.init_scale(scale, fname),
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def tp(
        self,
        *x: str,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
        remove_null_space_completely: bool = False,
    ) -> StrctTerm:
        r"""
        Thin plate spline.

        Parameters
        ----------
        *x
            Names of input variables (one or more).
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        penalty_order
            Order of the penalty. Quote from mgcv: "The default is to set this to the
            smallest value satisfying ``2*penalty_order > d+1`` where ``d`` is the
            number of covariates of the term."
        knots
            Knots used to set up the basis. If ``None`` (default), a set knots will be
            set up automatically.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.
        remove_null_space_completely
            If ``True``, the unpenalized part of the smooth, corresponding to the null
            space of the penalty matrix, is removed completely.

        See Also
        --------

        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

        References
        ----------
        - Wood, S.N. (2003) Thin-plate regression splines. Journal of the Royal
          Statistical Society (B) 65(1):95-114.
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.tp("x_nonlin", k=20)
        StrctTerm(name="tp(x_nonlin)")
        """
        basis = self.bases.tp(
            *x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
            remove_null_space_completely=remove_null_space_completely,
        )

        fname = self.names.fname("tp", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=self.init_scale(scale, fname),
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def ts(
        self,
        *x: str,
        k: int,
        scale: ScaleIG | lsl.Var | float | VarIGPrior | Literal["default"] = "default",
        inference: InferenceTypes | None | Literal["default"] = "default",
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        factor_scale: bool = False,
    ) -> StrctTerm:
        r"""
        Thin plate spline with additional null space penalty.

        Parameters
        ----------
        *x
            Names of input variables (one or more).
        k
            Number of (unconstrained) bases.
        scale
            Scale parameter passed to the coefficient prior, :attr:`.StrctTerm.scale`.

            - If you pass a ``float``, this will be taken as the constant value of
              the scale, and the scale will not be estimated as part of the model
              without further action.
            - If you pass a :class:`liesel.model.Var`, this will be used as the scale.
              Make sure to define the ``inference`` attribute of your custom
              scale variable (or a latent, transformed version of it).
            - If you pass a :class:`.VarIGPrior`, a scale variable will be set up for
              you using :class:`.ScaleIG`. This means, the scale will be
              :math:`\tau`, with an iverse Gamma prior on its square, i.e.
              :math:`\tau^2 \sim \operatorname{InverseGamma}(a, b)`, where a and b
              are taken from the :class:`.VarIGPrior` object. A fitting Gibbs kernel
              will be set up automatically to sample :math:`\tau^2` in this case,
              see :class:`.ScaleIG` for details.
        inference
            Inference specification for this term's coefficient.
            Note that this inference is only used for the coefficient variables
            of the terms created by this builder (:attr:`.StrctTerm.coef`), *not* for
            the scale variables (:attr:`.StrctTerm.scale`).
        penalty_order
            Order of the penalty. Quote from mgcv: "The default is to set this to the
            smallest value satisfying ``2*penalty_order > d+1`` where ``d`` is the
            number of covariates of the term."
        knots
            Knots used to set up the basis. If ``None`` (default), a set knots will be
            set up automatically.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        factor_scale
            Whether to factor out the scale in the prior for this term, turning it
            into a partially (or fully) standardized form. See
            :meth:`.StrctTerm.factor_scale` for details.
        remove_null_space_completely
            If ``True``, the unpenalized part of the smooth, corresponding to the null
            space of the penalty matrix, is removed completely.

        See Also
        --------

        .BasisBuilder : Initializes the basis and penalty.

        Notes
        -----

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

        References
        ----------
        - Wood, S.N. (2003) Thin-plate regression splines. Journal of the Royal
          Statistical Society (B) 65(1):95-114.
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> tb.ts("x_nonlin", k=20)
        StrctTerm(name="ts(x_nonlin)")
        """
        basis = self.bases.ts(
            *x,
            k=k,
            penalty_order=penalty_order,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name="B",
        )

        fname = self.names.fname("ts", basis.x.name)

        coef_name = self.names.beta(fname)
        term = StrctTerm(
            basis,
            penalty=basis.penalty,
            name=fname,
            scale=self.init_scale(scale, fname),
            inference=self._get_inference(inference),
            coef_name=coef_name,
        )
        if factor_scale:
            term.factor_scale()
        return term

    def _ta(
        self,
        *marginals: StrctTerm,
        common_scale: ScaleIG
        | lsl.Var
        | float
        | VarIGPrior
        | Literal["default"]
        | None = None,
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = gs.MCMCSpec(
            gs.HMCKernel
        ),
        include_main_effects: bool = False,
        _fname: str = "ta",
    ) -> StrctTensorProdTerm:
        r"""
        General anisotropic tensor product term.

        .. warning::
            This method remove any default gibbs samplers and replace them with
            ``scales_inference`` on log level, since the full conditional for the
            variance parameters is not known in closed form for an anisotropic
            tensor product.

        Parameters
        ----------
        *marginals
            Marginal terms, subclasses of :class:`.StrctTerm`.
        common_scale
            A single, common scale to cover all marginal dimensions, resulting in an
            isotropic tensor product. This mean setting
            :math:`\tau^2_1 = \\dots = \tau^2_M = \tau^2` for all marginal smooths
            in the notation used in :class:`.StrctTensorProdTerm`.
        inference
            Inference specification for this term's coefficient.
        scales_inference
            If ``"default"``, uses the default inference passed to the TermBuilder
            upon initialization.
        include_main_effects
            If ``True``, the marginal terms will be added to this term's value.

        """
        inputs = ",".join(
            list(StrctTensorProdTerm._input_obs([t.basis for t in marginals]))
        )
        fname = self.names.create(f"{_fname}(" + inputs + ")")
        basis_name = self.names.create("B(" + inputs + ")")
        coef_name = self.names.beta(fname)

        if common_scale is not None and not isinstance(common_scale, float):
            common_scale = self.init_scale(common_scale, fname)
            _biject_and_replace_star_gibbs_with(
                common_scale, self._get_inference(scales_inference)
            )
        elif common_scale is not None:
            common_scale = self.init_scale(common_scale, fname)

        term = StrctTensorProdTerm(
            *marginals,
            common_scale=common_scale,
            name=fname,
            inference=self._get_inference(inference),
            coef_name=coef_name,
            include_main_effects=include_main_effects,
            basis_name=basis_name,
        )

        if not common_scale:
            for scale in term.scales:
                if not isinstance(scale, lsl.Var):
                    raise TypeError(
                        f"Expected scale to be a liesel.model.Var, got {type(scale)}"
                    )
                _biject_and_replace_star_gibbs_with(
                    scale, self._get_inference(scales_inference)
                )

        return term

    def tx(
        self,
        *marginals: StrctTerm,
        common_scale: ScaleIG
        | lsl.Var
        | float
        | VarIGPrior
        | Literal["default"]
        | None = None,
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = gs.MCMCSpec(
            gs.HMCKernel
        ),
    ) -> StrctTensorProdTerm:
        r"""
        General anisotropic tensor product interaction term without main effects.

        Includes only the tensor product interaction. Corresponds to ``mgcv::ti``.

        .. warning::
            This method remove any default gibbs samplers and replace them with
            ``scales_inference`` on log level, since the full conditional for the
            variance parameters is not known in closed form for an anisotropic
            tensor product.

        Parameters
        ----------
        *marginals
            Marginal terms, subclasses of :class:`.StrctTerm`.
        common_scale
            A single, common scale to cover all marginal dimensions, resulting in an
            isotropic tensor product. This mean setting
            :math:`\tau^2_1 = \dots = \tau^2_M = \tau^2` for all marginal smooths
            in the notation used in :class:`.StrctTensorProdTerm`.
        inference
            Inference specification for this term's coefficient.
        scales_inference
            If ``"default"``, uses the default inference passed to the TermBuilder
            upon initialization.

        See Also
        --------
        .StrctTensorProdTerm : The term class returned by this method; includes further
          details.
        .tf : Full tensor product, including main effects.

        Examples
        --------

        Using only the interaction term:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)

        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        >>> ps1 = tb.ps("x_nonlin", k=7)
        >>> ps2 = tb.ps("x_lin", k=7)

        >>> pred += tb.tx(ps1, ps2)
        >>> pred.terms
        {'tx(x_nonlin,x_lin)': StrctTensorProdTerm(name="tx(x_nonlin,x_lin)")}

        .. rubric:: Anova decomposition

        Including the main effects (this corresponds to :meth:`.tf`):

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)

        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        >>> ps1 = tb.ps("x_nonlin", k=7)
        >>> ps2 = tb.ps("x_lin", k=7)

        >>> pred += ps1, ps2, tb.tx(ps1, ps2)
        >>> len(pred.terms)
        3

        .. rubric:: Isotropic tensor product

        >>> import liesel_gam as gam
        >>> import tensorflow_probability.substrates.jax.bijectors as tfb
        >>> import tensorflow_probability.substrates.jax.distributions as tfd

        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        Marginal smooths:

        >>> ps1 = tb.ps("x_nonlin", k=7)
        >>> ps2 = tb.ps("x_lin", k=7)

        Initializing the scale variable:

        >>> scale = lsl.Var.new_param(
        ...     1.0,
        ...     distribution=lsl.Dist(tfd.HalfNormal, scale=20.0),
        ...     name="{x}",
        ... )
        >>> log_scale = scale.transform(
        ...     tfb.Exp(),
        ...     inference=gs.MCMCSpec(gs.HMCKernel),
        ...     name="ln({x})",
        ... )

        Initializing the interaction term:

        >>> tx1 = tb.tx(ps1, ps2, common_scale=scale)

        The :attr:`.StrctTensorProdTerm.scales` list now contains the same scale twice,
        leading to an isotropic tensor product.

        >>> tx1.scales[0]
        Var(name="$\tau_{tx(x_nonlin,x_lin)}$")
        >>> tx1.scales[1]
        Var(name="$\tau_{tx(x_nonlin,x_lin)}$")


        .. rubric:: Marginals with different dimensions

        In the Anova decomposition of a tensor product, you can supply marginals with
        more bases than the marginals used in the interaction term. This can be
        helpful for managing the curse of dimensionality, as the parameter
        count in the interaction term grows rapidly.

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)
        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        Independent marginal terms:

        >>> ps1 = tb.ps("x_nonlin", k=20)
        >>> ps2 = tb.ps("x_lin", k=20)
        >>> pred += ps1, ps2

        Adding an interaction term using marginals with fewer bases:

        >>> pred += tb.tx(
        ...     tb.ps("x_nonlin", k=7),
        ...     tb.ps("x_lin", k=7),
        ... )
        >>> len(pred.terms)
        3


        Note that this term has four variance parameters: Two of them govern the
        independent marginals, and another two govern the interaction term. This
        added flexibility when using :meth:`.tx` is a further difference to
        :meth:`.tf`.

        .. rubric:: Three-dimensional interaction

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)

        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        >>> ps1 = tb.ps("x_nonlin", k=20)
        >>> ps2 = tb.ps("x_lin", k=20)
        >>> ps3 = tb.ps("x", k=20)

        >>> pred += ps1, ps2, ps3

        >>> pred += tb.tx(
        ...     tb.ps("x_nonlin", k=7),
        ...     tb.ps("x_lin", k=7),
        ...     tb.ps("x", k=7),
        ... )

        >>> len(pred.terms)
        4
        >>> pred.terms["tx(x_nonlin,x_lin,x)"].coef.value.shape
        (216,)
        """
        return self._ta(
            *marginals,
            common_scale=common_scale,
            inference=self._get_inference(inference),
            scales_inference=scales_inference,
            include_main_effects=False,
            _fname="tx",
        )

    def tf(
        self,
        *marginals: StrctTerm,
        common_scale: ScaleIG
        | lsl.Var
        | float
        | VarIGPrior
        | Literal["default"]
        | None = None,
        inference: InferenceTypes | None | Literal["default"] = "default",
        scales_inference: InferenceTypes | None | Literal["default"] = gs.MCMCSpec(
            gs.HMCKernel
        ),
    ) -> StrctTensorProdTerm:
        r"""
        General full anisotropic tensor product term, including main effects and
        interction.

        Corresponds to ``mgcv::te``.

        .. warning::
            This method remove any default gibbs samplers and replace them with
            ``scales_inference`` on log level, since the full conditional for the
            variance parameters is not known in closed form for an anisotropic
            tensor product.

        Parameters
        ----------
        *marginals
            Marginal terms, subclasses of :class:`.StrctTerm`.
        common_scale
            A single, common scale to cover all marginal dimensions, resulting in an
            isotropic tensor product. This mean setting
            :math:`\tau^2_1 = \dots = \tau^2_M = \tau^2` for all marginal smooths
            in the notation used in :class:`.StrctTensorProdTerm`.
        inference
            Inference specification for this term's coefficient.
        scales_inference
            If ``"default"``, uses the default inference passed to the TermBuilder
            upon initialization.

        See Also
        --------
        .StrctTensorProdTerm : The term class returned by this method; includes further
          details.
        .tx : Tensor product interaction term, without main effects.

        Examples
        --------

        Using only the interaction term:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)

        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        >>> ps1 = tb.ps("x_nonlin", k=7)
        >>> ps2 = tb.ps("x_lin", k=7)

        >>> pred += tb.tf(ps1, ps2)
        >>> pred.terms
        {'tf(x_nonlin,x_lin)': StrctTensorProdTerm(name="tf(x_nonlin,x_lin)")}

        To illustrate the difference to :meth:`.tx`, consider this example, which
        is practically equivalent:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)

        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        >>> ps1 = tb.ps("x_nonlin", k=7)
        >>> ps2 = tb.ps("x_lin", k=7)

        >>> pred += ps1, ps2, tb.tx(ps1, ps2)
        >>> len(pred.terms)
        3

        .. rubric:: Three-dimensional tensor product

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(100)

        >>> tb = gam.TermBuilder.from_df(df)
        >>> pred = gam.AdditivePredictor(name="loc")

        >>> pred += tb.tf(
        ...     tb.ps("x_nonlin", k=7),
        ...     tb.ps("x_lin", k=7),
        ...     tb.ps("x", k=7),
        ... )

        >>> len(pred.terms)
        1
        >>> pred.terms["tf(x_nonlin,x_lin,x)"].coef.value.shape
        (216,)
        """
        return self._ta(
            *marginals,
            common_scale=common_scale,
            inference=self._get_inference(inference),
            scales_inference=scales_inference,
            include_main_effects=True,
            _fname="tf",
        )


def _find_parameter(var: lsl.Var) -> lsl.Var:
    """
    Intended for the following use case: 'var' is a parameter that may be a
    weak transformation of a strong latent parameter, we want to find this
    strong latent parameter.

    Returns the strong latent parameter, if it can be determined unambiguously.
    """
    if var.strong and var.parameter:
        return var

    with TemporaryModel(var, to_float32=False, silent=True) as model:
        params = model.parameters
        if not params:
            raise ValueError(f"No parameter found in the graph of {var}.")
        if len(params) > 1:
            raise ValueError(
                f"In the graph of {var}, there are {len(params)} parameters, "
                "so we cannot return a unique parameter."
            )
        param = list(model.parameters.values())[0]

    return param


def _biject_and_replace_star_gibbs_with(
    var: lsl.Var, inference: InferenceTypes | None
) -> lsl.Var:
    """
    If var is a ScaleIG, it is the square root of a variance
    parameter that may have a default Gibbs kernel. This function removes any such
    Gibbs kernel and then transforms the variance parameter using the default event
    space bijector and sets the inference to the 'inference' supplied to the function.
    """
    param = _find_parameter(var)
    if param.inference is not None:
        if isinstance(param.inference, gs.MCMCSpec):
            try:
                is_star_gibbs = param.inference.kernel.__name__ == "StarVarianceGibbs"  # type: ignore
                if not is_star_gibbs:
                    return var
            except AttributeError:
                # in this case, we assume that the inference has been set intentionally
                # so we don't change anything
                return var
        else:
            # in this case, we assume that the inference has been set intentionally
            # so we don't change anything
            return var
    if param.name:
        trafo_name = "ln(" + param.name + ")"
    else:
        trafo_name = None
    transformed = param.transform(
        bijector=tfb.Exp(), inference=inference, name=trafo_name
    )
    if trafo_name is None:
        transformed.name = ""
    return var


def _has_star_gibbs(var: lsl.Var) -> bool:
    try:
        param = _find_parameter(var)
    except ValueError:
        return False
    if param.inference is None:
        # no inference means no StarVarianceGibbs
        return False

    inferences = []
    if isinstance(param.inference, gs.MCMCSpec):
        inferences.append(param.inference)
    elif isinstance(param.inference, Mapping):
        try:
            for v in param.inference.values():
                if isinstance(v, gs.MCMCSpec):
                    inferences.append(v)
        except Exception as e:
            raise TypeError(
                f"Could not handle type {type(param.inference)}, expected "
                "liesel.goose.MCMCSpec or dict."
            ) from e
    else:
        raise TypeError(
            f"Could not handle type {type(param.inference)}, expected "
            "liesel.goose.MCMCSpec or dict."
        )

    if not inferences:
        # no gs.MCMCSpecs present, so there cannot be StarVarianceGibbs
        return False

    for inference in inferences:
        try:
            is_star_gibbs = inference.kernel.__name__ == "StarVarianceGibbs"  # type: ignore
            if is_star_gibbs:
                return True  # if we find any StarVarianceGibbs, return True
        except Exception:
            # very liberal about errors here
            pass

    # by this point, we did not find any StarVarianceGibbs
    return False


def _format_name(var: lsl.Var, fill: str) -> lsl.Var:
    with TemporaryModel(var, to_float32=False, silent=True) as model:
        nodes = dict(model.nodes)
        vars_ = dict(model.vars)

    nodes_and_vars = nodes | vars_
    for node in nodes_and_vars.values():
        node.name = node.name.format(name=fill, x=fill)
        if "$" in node.name:
            node.name = node.name.replace("$", "")
            node.name = "$" + node.name + "$"

    if not var.name:
        var.name = fill

    return var
