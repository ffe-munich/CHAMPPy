champpy.ChargingPlotter
-----------------------

The :class:`~champpy.ChargingPlotter` is a factory class to visualize synthetic charging profiles in CHAMPPy.
Different plots are created with the Python library :mod:`plotly` and merged in a html file. 
The :class:`~champpy.ChargingPlotter` can be used to analyze the generated synthetic charging profiles and to compare different charging profiles.

The generated html file contains the following plots:

- 📊 Bar charts of charging characteristics: daily driving consumption, daily charging hours, daily charging energy, daily connected hours
- 📈 Total load profile of the fleet over the week 

**Basic workflow:**

1. Initialize the plotter :class:`~champpy.ChargingPlotter` with user parameters :class:`~champpy.UserParamsChargingPlotter`
2. Call :meth:`~champpy.ChargingPlotter.plot_charging_profiles` to create plots of :class:`~champpy.ChargingProfiles`.
3. Analyze the generated plots in html format, that opens automatically after the plot is created.

.. autoclass:: champpy.ChargingPlotter
   :members:
   :undoc-members:
   :inherited-members:
   :member-order: bysource

.. autoclass:: champpy.UserParamsChargingPlotter
   :members:
   :undoc-members:
   :inherited-members:
   :member-order: bysource