import logging
import pandas as pd
import numpy as np

from rich.progress import track
from pandera.pandas import Field as pa_Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import field_validator, Field as pydantic_Field
from dataclasses import dataclass, field

from champpy.core.mobility.mobility_components import Vehicles, VehiclesSchema, Clusters
from champpy.core.mobility.mobility_data import MobArray, MobProfiles


@pydantic_dataclass
class UserParamsChargingModel:
    """
    Data class to define user parameters for the charging profile generation.

    This configuration class defines electric vehicle properties and charging
    behavior used by :class:`ChargingModel`.

    Examples
    --------
    >>> params = UserParamsChargingModel(
    ...     energy_consumption_kwh_per_km=[0.2],
    ...     battery_capacity_kwh=[60.0],
    ...     charging_power_max_kw=[11.0],
    ...     charging_locations=[1, 2],
    ...     temp_res=0.25,
    ... )
    """

    energy_consumption_kwh_per_km: list[float] = field(default_factory=lambda: [0.2])
    """Energy consumption in kWh/km per vehicle or as a scalar list."""

    battery_capacity_kwh: list[float] = field(default_factory=lambda: [50.0])
    """Battery capacity in kWh per vehicle or as a scalar list."""

    charging_power_max_kw: list[float] = field(default_factory=lambda: [7.0])
    """Maximum charging power in kW per vehicle or as a scalar list."""

    efficiency_charging: list[float] = field(default_factory=lambda: [0.9])
    """Charging efficiency in the interval $(0, 1]$ per vehicle or as a scalar list."""

    soc_min: list[float] = field(default_factory=lambda: [0.1])
    """Minimum state of charge (SoC) in the interval $[0, 1)$ per vehicle."""

    soc_min_dep: list[float] = field(default_factory=lambda: [0.8])
    """Minimum SoC required at departure in the interval $(0, 1]$ per vehicle."""

    soc_initial: float = pydantic_Field(ge=0, le=1, default=1)
    """Initial SoC at simulation start in the interval $[0, 1]$."""

    distribute_energy_consumption: bool = True
    """If ``True``, distribute trip energy over all trip time steps."""

    charging_locations: list[int] = field(default_factory=lambda: [1])
    """Location IDs where charging is allowed."""

    temp_res: float = pydantic_Field(ge=1 / 60, default=0.25)
    """Temporal resolution in hours used for charging simulation."""

    @field_validator("energy_consumption_kwh_per_km")
    def check_energy_consumption(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("energy_consumption_kwh_per_km must be non-negative")
        return v

    @field_validator("battery_capacity_kwh")
    def check_battery_capacity(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError("battery_capacity_kwh must be positive")
        return v

    @field_validator("charging_power_max_kw")
    def check_charging_power(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError("charging_power_max_kw must be positive")
        return v

    @field_validator("efficiency_charging")
    def check_efficiency_charging(cls, v):
        if not (0 < v[0] <= 1):
            raise ValueError("efficiency_charging must be in (0, 1]")
        return v

    @field_validator("soc_min")
    def check_soc_min(cls, v):
        if not (0 <= v[0] < 1):
            raise ValueError("soc_min must be in [0, 1)")
        return v

    @field_validator("soc_min_dep")
    def check_soc_min_dep(cls, v):
        if not (0 < v[0] <= 1):
            raise ValueError("soc_min_dep must be in (0, 1]")
        return v

    @field_validator("charging_locations")
    def check_charging_locations(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("charging_locations must be non-negative")
        return v

    @field_validator("temp_res")
    def check_temp_res(cls, v):
        if v <= 0:
            raise ValueError("temp_res must be positive")
        return v


class EletricVehiclesSchema(VehiclesSchema):
    """
    Schema for electric vehicles, extending the VehiclesSchema with additional parameters
    needed for charging profile calculations.
    """

    energy_consumption_kwh_per_km: float = pa_Field(ge=0, coerce=True)
    battery_capacity_kwh: float = pa_Field(ge=0, coerce=True)
    charging_power_max_kw: float = pa_Field(ge=0, coerce=True)
    efficiency_charging: float = pa_Field(ge=0, coerce=True)
    soc_min: float = pa_Field(ge=0, coerce=True)
    soc_min_dep: float = pa_Field(ge=0, coerce=True)


class ElectricVehicles(Vehicles):
    """Electric vehicle component extending :class:`Vehicles`.

    This class is based on the vehicles contained in :class:`~champpy.MobProfiles`
    and adds charging-related parameters required for charging profile calculations.

    The DataFrame (accessible via the ``df`` property) contains all columns from
    :class:`Vehicles` plus additional charging-related parameters:

    .. list-table::
       :header-rows: 1

       * - Column
         - Type
         - Description
       * - id_vehicle
         - :class:`int`
         - Vehicle identifier.
       * - first_day
         - :class:`pandas.Timestamp`
         - First recorded day of the vehicle.
       * - last_day
         - :class:`pandas.Timestamp`
         - Last recorded day of the vehicle.
       * - id_cluster
         - :class:`int`
         - Cluster assignment.
       * - first_loc
         - :class:`int`
         - First location of the vehicle.
       * - energy_consumption_kwh_per_km
         - :class:`float`
         - Energy consumption per kilometer in kWh/km.
       * - battery_capacity_kwh
         - :class:`float`
         - Battery capacity in kWh.
       * - charging_power_max_kw
         - :class:`float`
         - Maximum charging power in kW.
       * - efficiency_charging
         - :class:`float`
         - Charging efficiency (0-1).
       * - soc_min
         - :class:`float`
         - Minimum state of charge (0-1).
       * - soc_min_dep
         - :class:`float`
         - Minimum state of charge at departure (0-1).
    """

    _schema = EletricVehiclesSchema

    def __init__(self, vehicles_data: Vehicles, user_params: UserParamsChargingModel):
        vehicles_df = vehicles_data.df
        num_vehicles = len(vehicles_df)
        param_list = [
            "energy_consumption_kwh_per_km",
            "battery_capacity_kwh",
            "charging_power_max_kw",
            "efficiency_charging",
            "soc_min",
            "soc_min_dep",
        ]
        for param in param_list:
            if not hasattr(user_params, param):
                raise ValueError(f"user_params must have attribute '{param}'")
            vehicles_df[param] = self._convert_userparam_to_arrays(getattr(user_params, param), num_vehicles)
        super().__init__(vehicles_df)

    def _convert_userparam_to_arrays(self, user_param: list[float], num_vehicles: int) -> np.ndarray:
        """Convert user parameters to arrays matching the number of vehicles."""
        arr = np.asarray(user_param)
        if arr.size == 1:
            return np.full(num_vehicles, arr.item())
        elif arr.size == num_vehicles:
            return arr
        else:
            raise ValueError("user_param must have length 1 or number of vehicles")

    def generate_vehicles_from_logbooks(self, logbooks):
        """Not allowed for ElectricVehicles. Only implemented for Vehicles class ."""
        raise AttributeError("generate_vehicles_from_logbooks is not available for ElectricVehicles class.")

    def delete_vehicles(self, id_vehicle):
        """Not allowed for ElectricVehicles. Only implemented for Vehicles class ."""
        raise AttributeError("delete_vehicles is not available for ElectricVehicles class.")

    def set_first_loc_from_logbooks(self, logbooks):
        """Not allowed for ElectricVehicles. Only implemented for Vehicles class ."""
        raise AttributeError("set_first_loc_from_logbooks is not available for ElectricVehicles class.")

    def update_vehicles(self, input_df):
        """Not allowed for ElectricVehicles. Only implemented for Vehicles class ."""
        raise AttributeError("update_vehicles is not available for ElectricVehicles class.")

    def add_vehicles(self, input_df):
        """Not allowed for ElectricVehicles. Only implemented for Vehicles class ."""
        raise AttributeError("add_vehicles is not available for ElectricVehicles class.")


@dataclass
class ChargingArray:
    """Class to hold charging profiles as numpy arrays."""

    datetime: np.ndarray = field(default_factory=lambda: np.array([]))
    id_vehicle: np.ndarray = field(default_factory=lambda: np.array([]))
    power_charging_kw: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_stored_kwh: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_missing_kwh: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_consumption_kwh: np.ndarray = field(default_factory=lambda: np.array([]))
    connected: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


class ChargingTimeseries:
    """Class to hold charging profiles as a DataFrame.

    Each row corresponds to a time step and a vehicle and contains information
    about charging status, energy consumption, energy stored, charging power,
    and missing energy.

    The DataFrame (accessible via the `df` property) contains the following columns:

    .. list-table::
       :header-rows: 1

       * - Column
         - Type
         - Description
       * - id_vehicle
         - :class:`int`
         - Vehicle identifier.
       * - datetime
         - :class:`pandas.Timestamp`
         - Date and time of the time step.
       * - connected
         - :class:`bool`
         - Boolean indicating whether the vehicle is connected to a charging station at the given time step.
       * - energy_consumption_kwh
         - :class:`float`
         - Energy consumption in kWh at the given time step.
       * - energy_stored_kwh
         - :class:`float`
         - Energy stored in the battery in kWh at the given time step.
       * - power_charging_kw
         - :class:`float`
         - Charging power in kW at the given time step.
       * - energy_missing_kwh
         - :class:`float`
         - Energy in kWh that is missing at the given time step (i.e., energy that should have been charged but was not because the vehicle was not connected).
    """

    def __init__(self, charging_array: ChargingArray):

        if isinstance(charging_array, ChargingArray) is False:
            raise ValueError("charging_array must be an instance of ChargingArray class.")

        self._df = pd.DataFrame(
            {
                "id_vehicle": np.tile(charging_array.id_vehicle, len(charging_array.datetime)),
                "datetime": np.repeat(charging_array.datetime, len(charging_array.id_vehicle)),
                "connected": charging_array.connected.ravel(),
                "energy_consumption_kwh": charging_array.energy_consumption_kwh.ravel(),
                "energy_stored_kwh": charging_array.energy_stored_kwh.ravel(),
                "power_charging_kw": charging_array.power_charging_kw.ravel(),
                "energy_missing_kwh": charging_array.energy_missing_kwh.ravel(),
            }
        )

        # Sort df by id_vehicle and datetime
        self._df = self._df.sort_values(by=["id_vehicle", "datetime"]).reset_index(drop=True)

    @property
    def df(self):
        """Return a copy of the DataFrame containing the charging timeseries data."""
        return self._df.copy()


class ChargingProfiles:
    """
    Wrapper class for charging data in the champpy framework.
    It contains instances of the following classes: :class:`ChargingTimeseries`, :class:`ElectricVehicles`, :class:`Clusters`.
    This class is genereated by the :class:`ChargingModel` class. Don't instantiate it directly.

    Attributes
    ----------
    charging_timeseries: :class:`ChargingTimeseries`
            Contains the charging data for each vehicle over time.
    vehicles: :class:`ElectricVehicles`
            Contains vehicle-specific data about each vehicle,
            such as its first and last day of activity, cluster assignment, battery capacity, max. charging power etc.
            It is connected to logbooks via `id_vehicle`.
    clusters: :class:`Clusters`
            Describes the clusters defined in vehicles. It is connected to vehicles via `id_cluster`.
            It provides a label for each cluster.
    """

    def __init__(
        self,
        charging_array: ChargingArray,
        vehicles: ElectricVehicles,
        clusters: Clusters,
    ):
        self.charging_timeseries = ChargingTimeseries(charging_array)
        self.vehicles = vehicles
        self.clusters = clusters


class ChargingModel:
    """
    Model for generating charging profiles from mobility data and user parameters.

    The model uses an algorithm that iterates over each time step and each vehicle to determine
    the charging status, energy consumption, energy stored, and missing energy based on the mobility data
    and user parameters. The resulting charging profiles are stored in a :class:`ChargingProfiles` object.

    Parameters
    ----------
    mob_profiles : :class:`~champpy.MobProfiles`
        Mobility profiles containing vehicle movement data and temporal information.

    Examples
    --------
    Generate charging profiles for synthetic mobility data:

    .. code-block:: python

        from champpy import MobModel, ChargingModel, UserParamsChargingModel

        # Generate synthetic mobility profiles
        mob_model = MobModel(model_params)
        mob_profiles = mob_model.generate_mob_profiles(num_vehicles=100, days=7)

        # Create charging model
        charging_model = ChargingModel(mob_profiles)

        # Configure charging parameters
        charging_params = UserParamsChargingModel(
            energy_consumption_kwh_per_km=[0.2],
            battery_capacity_kwh=[60.0],
            charging_power_max_kw=[7.0],
            charging_locations=[1]  # Charge at locations 1
        )

        # Generate charging profiles
        charging_profiles = charging_model.generate_charging_profiles(charging_params)
    """

    def __init__(self, mob_profiles: MobProfiles):
        """
        Initialize charging model from mobility profiles.

        Parameters
        ----------
        mob_profiles : :class:`~champpy.MobProfiles`
            Mobility profiles containing logbooks, vehicles, and clusters.
        """
        self._mob_arrays = MobArray(mob_profiles)
        self._temp_res = mob_profiles.logbooks.temp_res  # in hours
        self._vehicles = mob_profiles.vehicles
        self._clusters = mob_profiles.clusters
        self._number_steps = len(self._mob_arrays.datetime)
        self._num_vehicles = mob_profiles.vehicles.number
        self._dt = self._mob_arrays.datetime
        self._id_vehicle = self._mob_arrays.id_vehicle

    def generate_charging_profiles(self, user_params: UserParamsChargingModel) -> ChargingProfiles:
        """
        Generate charging profiles from mobility data and charging parameters.

        Parameters
        ----------
        user_params : UserParamsChargingModel
            User-defined charging model parameters.

        Returns
        -------
        ChargingProfiles
            Generated charging profiles including timeseries, electric vehicles,
            and clusters.
        """
        # Define vehicles
        vehicles = ElectricVehicles(self._vehicles, user_params)

        # Predefine necessary variables
        self._predefine_vars(vehicles, user_params)

        # Determine connection status array
        connected_array = np.isin(self._mob_arrays.location, user_params.charging_locations)
        self._charging_array.connected = connected_array

        # Determien energy consumption
        if user_params.distribute_energy_consumption:
            energy_consumption_array = self._mob_arrays.distance_distributed * self._energy_cons_array
        else:
            energy_consumption_array = self._mob_arrays.distance * self._energy_cons_array
        self._charging_array.energy_consumption_kwh = energy_consumption_array

        # Determine minimum soc departure from required driving energy and min soc at departure
        soc_min_departure_array = self._determine_min_soc_departure()
        energy_min_departure_array = soc_min_departure_array * self._battery_capacity_array

        mssg = "Generating charging profiles based on mobility data and user parameters..."
        logging.info(mssg)

        # Loop over timesteps and evs
        for t in track(range(0, self._number_steps), description="Generating charging profiles:"):
            # Get stored energy at beginning of timestep
            if t != 0:
                # set stored energy for current timestep to the one at the previous timestep
                stored_energy = self._charging_array.energy_stored_kwh[t - 1, :]
            else:
                # at t=0 use initial stored energy
                stored_energy = self._soc_initial_array * self._battery_capacity_array

            # discharge battery based on energy consumption
            stored_energy = stored_energy - energy_consumption_array[t, :]

            # Check which vehicles are connected
            mask_con = connected_array[t, :] & (energy_min_departure_array[t, :] > stored_energy)

            # Charge connected vehicles
            energy_to_charge = energy_min_departure_array[t, mask_con] - stored_energy[mask_con]
            necessary_power = energy_to_charge / (self._efficiency_charging_array[mask_con] * user_params.temp_res)
            maximum_power = self._charging_power_max_array[mask_con]
            charging_power = np.minimum(maximum_power, necessary_power)
            energy_charged = charging_power * self._efficiency_charging_array[mask_con] * user_params.temp_res

            # Determine missing energy of not connected vehicles
            energy_min = self._soc_min_array[~mask_con] * self._battery_capacity_array[~mask_con]
            missing_energy = np.maximum(0, energy_min - stored_energy[~mask_con])
            stored_energy[~mask_con] = stored_energy[~mask_con] + missing_energy
            # TODO: Missing energy soll pro Fahrt nur einmal berechnet werden, nicht pro Zeitschritt

            # save variables
            self._charging_array.power_charging_kw[t, mask_con] = charging_power
            self._charging_array.energy_stored_kwh[t, :] = stored_energy
            self._charging_array.energy_stored_kwh[t, mask_con] += energy_charged
            self._charging_array.energy_missing_kwh[t, ~mask_con] = missing_energy

        # Create charging_profiles
        charging_profiles = ChargingProfiles(self._charging_array, vehicles=vehicles, clusters=self._clusters)
        return charging_profiles

    def _predefine_vars(self, vehicles: ElectricVehicles, user_params: UserParamsChargingModel):
        """Predefine variables needed for charging profile calculations."""

        # Dictionary: UserParams-Name -> Target-Array-Name
        param_map = {
            "energy_consumption_kwh_per_km": "_energy_cons_array",
            "battery_capacity_kwh": "_battery_capacity_array",
            "charging_power_max_kw": "_charging_power_max_array",
            "efficiency_charging": "_efficiency_charging_array",
            "soc_min": "_soc_min_array",
            "soc_min_dep": "_soc_min_dep_array",
        }
        for param_name, array_name in param_map.items():
            value = vehicles.df[param_name].to_numpy()
            setattr(self, array_name, value)

        self._soc_initial_array = np.full(self._num_vehicles, user_params.soc_initial)

        # Predefine charging data arrays
        self._charging_array = ChargingArray(
            datetime=self._dt,
            id_vehicle=self._id_vehicle,
            power_charging_kw=np.zeros((self._number_steps, self._num_vehicles)),
            energy_stored_kwh=np.zeros((self._number_steps, self._num_vehicles)),
            energy_missing_kwh=np.zeros((self._number_steps, self._num_vehicles)),
        )

    def _determine_min_soc_departure(self) -> np.ndarray:
        """Determine minimum state of charge required for departure."""
        soc_min_departure_raw = self._mob_arrays.distance * self._energy_cons_array / self._battery_capacity_array
        soc_min_departure_array = pd.DataFrame(soc_min_departure_raw).replace(0, np.nan).bfill().to_numpy()
        soc_min_departure_array = np.nan_to_num(soc_min_departure_array, nan=1)
        # Minimum soc at departure
        min_dep = np.broadcast_to(self._soc_min_dep_array, soc_min_departure_array.shape)
        soc_min_departure_array = np.maximum(soc_min_departure_array, min_dep)
        # Maximum soc at departure
        max_dep = 1.0
        soc_min_departure_array = np.minimum(soc_min_departure_array, max_dep)
        return soc_min_departure_array
