champpy.ChargingProfiles
------------------------

The :class:`~champpy.ChargingProfiles` class is a wrapper class that integrates three interconnected data components for managing charging profiles. 
These three components are instances of :class:`~champpy.ChargingTimeseries`, :class:`~champpy.ElectricVehicles`, and :class:`~champpy.Clusters`.
Each component contains a :class:`~pandas.DataFrame` that holds the data for that component.
The components are linked via IDs, ensuring data consistency and enabling seamless workflows for analyzing and modifying the data.
The structure is as follows:

.. code-block:: text

    ChargingProfiles
    ├── charging_timeseries  # Charging data for each timestep 
    │   └── df               # DataFrame with one row for each timestep and vehicle
    ├── vehicles             # Information of the vehicles
    │   └── df               # DataFrame with one row for each vehicle 
    └── clusters             # Groups of vehicles with similar behaviour
        └── df               # DataFrame with one row for each cluster  

.. autoclass:: champpy.ChargingProfiles
   :members:
   :undoc-members:

.. autoclass:: champpy.ChargingTimeseries
   :members:
   :undoc-members:

.. autoclass:: champpy.ElectricVehicles
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: generate_vehicles_from_logbooks,delete_vehicles,set_first_loc_from_logbooks,update_vehicles,add_vehicles