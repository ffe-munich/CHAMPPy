champpy.ParamsLoader
----------------------

The :class:`~champpy.ParamsLoader` class is used to load existing model parameters :class:`~champpy.ModelParams` required as input for the mobility model :class:`~champpy.MobModel`.

**Basic workflow:**

1. Create an instance of the ParamsLoader class
2. Call :meth:`~champpy.ParamsLoader.load_info` to check what parameters are available
3. Select parameters by choosing the corresponding ``id_params``
4. Call :meth:`~champpy.ParamsLoader.load_params` with the selected ``id_params``

The model parameters :class:`~champpy.ModelParams` have the following structure:

.. code-block:: text

    ModelParams
    ├── df    # Dataframe holding the model parameters
    └── info  # Meta information about the parameters

To generate new model parameters, please use the :class:`~champpy.Parameterizer` class.

.. autoclass:: champpy.ParamsLoader
   :members:
   :undoc-members: