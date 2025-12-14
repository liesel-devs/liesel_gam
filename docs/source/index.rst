.. liesel-ptm documentation master file, created by
   sphinx-quickstart on Mon Jul  3 09:59:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bayesian Generalized Additive Models in Liesel
==============================================

Learn more in the paper:

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2024). Bayesian penalized transformation models: Structured additive location-scale regression for arbitrary conditional distributions (No. arXiv:2404.07440). arXiv. `https://doi.org/10.48550/arXiv.2404.07440 <https://doi.org/10.48550/arXiv.2404.07440>`_

Installation
------------

The library can be installed from PYPI:

.. code:: bash

    $ pip install liesel_gam



API Reference
-------------

.. rubric:: Model

.. autosummary::
    :toctree: generated
    :caption: API
    :nosignatures:

    ~liesel_gam.BasisBuilder
    ~liesel_gam.TermBuilder
    ~liesel_gam.Basis
    ~liesel_gam.StrctTerm


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
