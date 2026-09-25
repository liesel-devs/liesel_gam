.. _nb_large_data:

Large-data workflows
====================

.. toctree::
   :maxdepth: 1
   :glob:

   notebooks/large_data/*

.. _precomputed-batches:

Precompute basis rows
---------------------

When basis calculations are expensive, compute their matrices once and reuse
their rows during minibatch fitting. The full matrices must fit in memory.
This example requires a Liesel version with ``liesel.optim`` and support for
computed variables as split and batch data.

Assume you have a :class:`.TermBuilder` named ``tb``, a Liesel ``model`` built
from its terms, and a DataFrame ``data`` containing the observations to fit.
Here, all observations belong to training, and each data entry has one row per
observation.

.. code-block:: python

   import liesel.optim as opt
   import optax

   position = tb.registry.precomputed_position(model, data)
   split = opt.Split(axis_size=len(data)).split_position(position)

Build the split from this returned dictionary: it contains the matrix keys and
any covariates that must stay raw. ``LieselOptim`` then uses those same keys to
slice matching rows for every batch.

.. code-block:: python

   result = opt.LieselOptim(
       model,
       split=split,
       batch_size=128,
       optimizers=optax.adam(0.01),
       loss_monitor=opt.EmaTrainLossMonitor(effective_window=10),
       stopper=opt.Stopper(epochs=100, patience=10),
       seed=42,
   ).fit()

Choose a batch size no larger than the training set. The monitor smooths the
training loss across batches; ``result.position_min_monitor`` contains the
parameter values selected by that monitor.

Covariates shared with terms that still need raw inputs remain raw. See
:meth:`.DictRegistry.precomputed_position` for the selection rules. For
prediction on new covariates, use :meth:`.DictRegistry.observed_position`.
