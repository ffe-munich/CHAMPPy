
champpy.MobProfiles
--------------------

The :class:`~champpy.MobProfiles` class is a wrapper class that integrates four interconnected data components for managing mobility profiles. 
These four components are instances of :class:`~champpy.Logbooks`, :class:`~champpy.Vehicles`, :class:`~champpy.Clusters`, and :class:`~champpy.Locations`.
Each component contains a :class:`~pandas.DataFrame` that holds the data for that component.
The components are linked via IDs, ensuring data consistency and enabling seamless workflows for analyzing and modifying the data.
The structure is as follows:

.. code-block:: text

    MobProfiles
    ├── logbooks  # Journeys of the vehicles
    │   └── df    # DataFrame with one row for each journey
    ├── vehicles  # Information of the vehicles
    │   └── df    # DataFrame with one row for each vehicle
    ├── clusters  # Groups of vehicles with similar behaviour
    │   └── df    # DataFrame with one row for each cluster
    └── locations # Information of the locations distinguished
        └── df    # DataFrame with one row for each unique location

.. autoclass:: champpy.MobProfiles
   :members:
   :undoc-members:

.. autoclass:: champpy.Logbooks
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.Vehicles
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.Clusters
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: champpy.Locations
   :members:
   :undoc-members:
   :inherited-members:



