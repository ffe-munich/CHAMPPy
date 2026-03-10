champpy.MobModel
---------------------

The :class:`~champpy.MobModel` class is used to generate synthetic mobility profiles based on a trained model.
It uses a Markov chain process to model vehicle locations over time and beta distributions to model speed and distance for each journey.

The model is initialized with pre-calculated parameters (:class:`~champpy.ModelParams`) from the parameterization step.
After initialization, you can generate synthetic mobility profiles for a specified number of vehicles over a defined time period
using the :meth:`~champpy.MobModel.generate_mob_profiles` method.

**Basic workflow:**

1. Initialize the model :class:`~champpy.MobModel` with :class:`~champpy.ModelParams` (from :class:`~champpy.Parameterizer` or :class:`~champpy.ParamsLoader`)
2. Configure user parameters using :class:`~champpy.UserParamsMobModel`
3. Call :meth:`~champpy.MobModel.generate_mob_profiles` to generate synthetic mobility profiles
4. Access the generated :class:`~champpy.MobProfiles` instance

.. autoclass:: champpy.MobModel
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.UserParamsMobModel
   :members:
   :undoc-members:
   :inherited-members: