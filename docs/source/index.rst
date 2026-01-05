Bayesian Generalized Additive Models in Liesel
==============================================

.. include:: welcome.md
   :parser: myst_parser.sphinx_

Installation
------------

The library can be installed from PYPI:

.. code:: bash

    $ pip install liesel_gam

Demo Notebooks
---------------

This documentation contains some notebooks that demonstrate how to put the pieces
together. The API documentation further below then provides extensive information on all
the individual pieces.

- :ref:`nb_lin`
- :ref:`nb_uni`
- :ref:`nb_multi`
- :ref:`nb_composite`

Check out the demos on polynomial regression and on P-splines for additional
example code, for example on on posterior predictive sampling.


.. toctree::
   :hidden:
   :caption: Demo Notebooks
   :maxdepth: 1

   notebooks_lin
   notebooks_univariate
   notebooks_composite
   notebooks_multivariate


Relevant Literature
--------------------

Fahrmeier et al. (2013) is a textbook that introduces structured additive
regression concepts from the ground up. Wood (2017) is another seminal textbook on
generalized additive models. The R package mgcv provides many basis functions and
penalty matrices that we use in ``liesel_gam``.


- Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013). Regression—Models, methods and
  applications. Springer. https://doi.org/10.1007/978-3-642-34333-9
- Wood, S. N. (2017). Generalized additive models (2nd ed.). Chapman & Hall/CRC.
- R package mgcv: https://cran.r-project.org/web/packages/mgcv/index.html

The other references are seminal papers on structured additive distributional
regression.

- Kneib, T., Klein, N., Lang, S., & Umlauf, N. (2019). Modular regression—A Lego system
  for building structured additive distributional regression models with tensor product
  interactions. TEST, 28(1), 1–39. https://doi.org/10.1007/s11749-019-00631-z
- Umlauf, N., Klein, N., & Zeileis, A. (2018). Bamlss: Bayesian additive models for
  location, scale, and shape (and beyond). Journal of Computational and Graphical
  Statistics, 27(3), 612–627. https://doi.org/10.1080/10618600.2017.1407325
- Klein, N., Kneib, T., Lang, S., & Sohn, A. (2015). Bayesian structured additive
  distributional regression with an application to regional income inequality in
  Germany. The Annals of Applied Statistics, 9(2), 1024–1052.
  https://doi.org/10.1214/15-AOAS823



API Reference
-------------

High-level API
***************

.. autosummary::
    :toctree: generated
    :caption: High-level API
    :nosignatures:

    ~liesel_gam.AdditivePredictor
    ~liesel_gam.TermBuilder
    ~liesel_gam.BasisBuilder

Plots
***************

.. autosummary::
    :toctree: generated
    :caption: Plots
    :nosignatures:

    ~liesel_gam.plot_1d_smooth
    ~liesel_gam.plot_2d_smooth
    ~liesel_gam.plot_forest
    ~liesel_gam.plot_polys
    ~liesel_gam.plot_regions
    ~liesel_gam.plot_1d_smooth_clustered

Summary
***************

.. autosummary::
    :toctree: generated
    :caption: Summary
    :nosignatures:

    ~liesel_gam.summarise_1d_smooth
    ~liesel_gam.summarise_nd_smooth
    ~liesel_gam.summarise_lin
    ~liesel_gam.summarise_cluster
    ~liesel_gam.summarise_regions
    ~liesel_gam.summarise_1d_smooth_clustered
    ~liesel_gam.summarise_by_samples
    ~liesel_gam.polys_to_df


Bases
***************

.. autosummary::
    :toctree: generated
    :caption: Bases
    :nosignatures:

    ~liesel_gam.Basis
    ~liesel_gam.MRFBasis
    ~liesel_gam.LinBasis


Terms and Variables
***************

.. autosummary::
    :toctree: generated
    :caption: Terms
    :nosignatures:

    ~liesel_gam.StrctTerm
    ~liesel_gam.StrctTensorProdTerm
    ~liesel_gam.LinTerm
    ~liesel_gam.StrctLinTerm
    ~liesel_gam.LinMixin
    ~liesel_gam.IndexingTerm
    ~liesel_gam.RITerm
    ~liesel_gam.MRFTerm
    ~liesel_gam.BasisDot
    ~liesel_gam.ScaleIG
    ~liesel_gam.UserVar


Distribution
***************

.. autosummary::
    :toctree: generated
    :caption: Distribution
    :nosignatures:

    ~liesel_gam.MultivariateNormalSingular
    ~liesel_gam.MultivariateNormalStructured
    ~liesel_gam.StructuredPenaltyOperator


Other
***************

.. autosummary::
    :toctree: generated
    :caption: Other
    :nosignatures:

    ~liesel_gam.PandasRegistry
    ~liesel_gam.CategoryMapping
    ~liesel_gam.MRFSpec
    ~liesel_gam.NameManager
    ~liesel_gam.VarIGPrior
    ~liesel_gam.demo_data
    ~liesel_gam.demo_data_ta
    ~liesel_gam.LinearConstraintEVD

.. rubric:: In/Out

.. autosummary::
    :toctree: generated
    :caption: In/Out
    :nosignatures:

    ~liesel_gam.io.read_bnd
    ~liesel_gam.io.polygon_is_closed

Experimental
***************

The API of modules, classes and functions in the experimental module is less stable
than in other modules of ``liesel_gam``. If you depend on this, expect changes in the
future.

.. autosummary::
    :toctree: generated
    :caption: Experimental
    :nosignatures:

    ~liesel_gam.experimental.BSplineApprox


Acknowledgements and Funding
--------------------------------

We are
grateful to the `German Research Foundation (DFG) <https://www.dfg.de/en>`_ for funding the development
through grant 443179956.

.. image:: https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg
   :alt: University of Göttingen

.. image:: https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg
   :alt: Funded by DFG


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
