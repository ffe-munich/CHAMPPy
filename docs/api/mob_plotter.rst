champpy.MobPlotter
---------------------

The :class:`~champpy.MobPlotter` is a factory class to visualize synthetic mobility profiles in CHAMPPy.
Different plots are are created with the Python library :mod:`plotly` and merged in a html file. 
The :class:`~champpy.MobPlotter` can be used to analyze the generated synthetic mobility profiles and to compare different mobility profiles, e.g,, reference profiles vs. modelled profiles.

The generated html file contain the following plots:

- 📊 Bar charts of mobility characteristics: daily kilometrage, daily journey time, number of journeys per day
- 📈 Histogram of mobility characteristics per day, per vehicle and per journey: daily kilometrage, daily journey time, number of journeys per day. 
- 📍 Locations of the mobility profiles over the average week 

**Basic workflow:**

1. Initialize the plotter :class:`~champpy.MobPlotter` with user parameters :class:`~champpy.UserParamsMobPlotter`
2. Call :meth:`~champpy.MobPlotter.plot_mob_profiles` to create plots of :class:`~champpy.MobProfiles`.
3. Analyze the genertaed plots in html format, that opens automatically after the plot is created.

.. autoclass:: champpy.MobPlotter
   :members:
   :undoc-members:
   :inherited-members:
   :member-order: bysource

.. autoclass:: champpy.UserParamsMobPlotter
   :members:
   :undoc-members:
   :inherited-members:
   :member-order: bysource