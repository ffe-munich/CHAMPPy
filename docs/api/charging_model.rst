champpy.ChargingModel
----------------------

The :class:`~champpy.ChargingModel` class is used to generate charging profiles based on driving profiles and user parameters.

Before using the charging model, synthetic mobility profiles must be generated using the :class:`~champpy.MobModel` class.

**Basic workflow:**

1. Initialize the model :class:`~champpy.ChargingModel` with synthetic mobility profiles :class:`~champpy.MobProfiles` 
2. Configure charging parameters using :class:`~champpy.UserParamsChargingModel`
3. Call :meth:`~champpy.ChargingModel.generate_charging_profiles` to generate synthetic charging profiles
4. Access the generated :class:`~champpy.ChargingProfiles` instance

.. autoclass:: champpy.ChargingModel
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.UserParamsChargingModel
   :members:
   :exclude-members: check_energy_consumption, check_battery_capacity, check_charging_power, check_efficiency_charging, check_soc_min, check_soc_min_dep, check_charging_locations, check_temp_res