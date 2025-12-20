Bayesian Generalized Additive Models in Liesel
==============================================


Installation
------------

The library can be installed from PYPI:

.. code:: bash

    $ pip install liesel_gam

Demo Notebooks
---------------

- :ref:`nb_lin`
- :ref:`nb_uni`
- :ref:`nb_multi`
- :ref:`nb_composite`


.. toctree::
   :hidden:
   :caption: Demo Notebooks
   :maxdepth: 1

   notebooks_lin
   notebooks_univariate
   notebooks_composite
   notebooks_multivariate


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
