champpy.Parameterizer
----------------------

The :class:`~champpy.Parameterizer` class is the factory class to calculate the model parameters :class:`~champpy.ModelParams` required as input for the mobility model :class:`~champpy.MobModel`.
The :class:`~champpy.Parameterizer` uses cleaned reference mobility profiles :class:`~champpy.MobProfiles` and user parameters :class:`~champpy.UserParamsParameterizer` as inputs.

**Basic workflow:**

1. Initialize the parameterizer :class:`~champpy.Parameterizer` with user parameters :class:`~champpy.UserParamsParameterizer`.
2. Call :meth:`~champpy.Parameterizer.calculate_params` to calculate the model parameters :class:`~champpy.ModelParams` using the cleaned reference mobility profiles :class:`~champpy.MobProfiles` as input. 
3. The calculated :class:`~champpy.ModelParams` instance can be further used for :class:`~champpy.MobModel`.

The model parameters :class:`~champpy.ModelParams` have the following structure:

.. code-block:: text

    ModelParams
    ├── df    # Dataframe holding the model parameters
    └── info  # Meta information about the parameters

Besided the :class:`~champpy.Parameterizer`, existing parameters can be loaded using the :class:`~champpy.ModelParamsLoader` class.

.. autoclass:: champpy.Parameterizer
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.UserParamsParameterizer
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.ModelParams
   :members:
   :undoc-members:

.. autoclass:: champpy.ParamsInfo
   :members:
   :undoc-members: