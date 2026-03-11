champpy.MobProfilesCleaner
--------------------------

The :class:`~champpy.MobProfilesCleaner` class is used to clean and validate mobility profiles data.
It provides configurable limits for speed, duration, and distance, allowing for flexible data cleaning strategies.

The cleaner performs the following operations:

- Applies min/max limits to speed, duration, and distance values
- Removes outliers or caps values based on configured methods
- Cleans first/last journey locations to ensure plausible data
- Resamples data to a specified temporal resolution
- Provides detailed logging of all cleaning actions

**Basic workflow:**

1. Initialize the cleaner class :class:`~champpy.MobProfilesCleaner` with user parameters :class:`~champpy.UserParamsCleaning`
2. Call :meth:`~champpy.MobProfilesCleaner.clean_profiles` with the mobility profiles :class:`~champpy.MobProfiles` to be cleaned
3. Access the cleaned :class:`~champpy.MobProfiles` instance

.. autoclass:: champpy.MobProfilesCleaner
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.UserParamsCleaning
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.LimitConfig
   :members:
   :undoc-members:
