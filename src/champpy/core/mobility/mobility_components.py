import pandas as pd

try:
    import pandera.pandas as pa
except ModuleNotFoundError:
    # Fallback for older Python versions
    import pandera as pa
from pandera.typing import Series
import logging
from typing import Literal, Optional
from pydantic import validate_call
from abc import ABC
from champpy.utils.data_utils import Event

# Configure logger for this module
logger = logging.getLogger(__name__)


class BaseMobilityComponent(ABC):
    """Base class for mobility components: Logbooks, Vehicles, Clusters, Locations."""

    _schema = None  # Overridden in subclasses with specific Pandera schema

    def __init__(self, input_df: Optional[pd.DataFrame] = None, frozen: bool = False):
        """Initialize a BaseMobilityComponent instance."""
        self._frozen = frozen
        if input_df is not None:
            self.df = input_df
        else:
            self._df = None

    def __init_subclass__(cls):
        """Ensure subclasses define a _schema attribute."""
        super().__init_subclass__()
        if getattr(cls, "_schema", None) is None:
            raise NotImplementedError(f"{cls.__name__} must define a class attribute '_schema'")

    @property
    def df(self) -> pd.DataFrame:
        """Get a copy of the DataFrame of the data component. If the DataFrame is None, return an empty DataFrame with the correct schema."""
        if self._df is None:
            output_df = self._schema.example(size=0)
        else:
            output_df = self._df.copy()
        output_df = self._on_df_getter(output_df)  # Hook method for subclasses
        return output_df

    @df.setter
    def df(self, input_df: pd.DataFrame):
        """Set the DataFrame of the data component with validation."""
        self._check_frozen()
        self._df = self._prep_input_df(input_df)
        self._on_df_setter()  # Hook method for subclasses

    def _prep_input_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Hook method to prepare the input DataFrame. Can be overridden in subclasses."""
        output_df = self._schema.validate(input_df)
        return output_df

    def _on_df_setter(self):
        """Hook method called after setting the DataFrame. Can be overridden in subclasses."""
        pass

    def _on_df_getter(self, output_df: pd.DataFrame) -> pd.DataFrame:
        """Hook method called when getting the DataFrame. Can be overridden in subclasses."""
        return output_df

    def _del_rows_of_df(self, mask_delete: pd.Series) -> None:
        """Delete rows from the DataFrame based on a boolean mask."""
        if self._df is None or self._df.empty:
            return
        self._check_frozen()
        new_df = self._df.loc[~mask_delete].copy().reset_index(drop=True)
        self._df = self._schema.validate(new_df)

    def _update_rows_of_df(
        self, input_df: pd.DataFrame, index_cols: list[str], user_setter: bool = True, prefer_input: bool = False
    ) -> None:
        """Update rows in the DataFrame based on index columns.
        Parameters:
                input_df (pd.DataFrame): DataFrame with rows to update.
                index_cols (list[str]): List of columns to use as index for matching rows.
                user_setter (bool, default=True): If True, use the df setter for updating (with validation/hooks).
                prefer_input (bool, default=False): If True, prefer values from input_df when updating rows. If false, prefer existing values.
        """
        if self._df is None or self._df.empty:
            if prefer_input and user_setter:
                self.df = input_df  # use setter for validation and hooks
            elif prefer_input and not user_setter:
                self._df = self._prep_input_df(input_df)
            return
        self._check_frozen()
        input_df = self._prep_input_df(input_df)
        # Set index for efficient update
        existing_df = self._df.set_index(index_cols)
        input_df = input_df.set_index(index_cols)
        if prefer_input:
            # Update input rows with values from existing_df, prefering input values
            input_df.update(existing_df)
            new_df = input_df
        else:
            # Update existing rows with values from input_df, prefering existing values
            existing_df.update(input_df)
            new_df = existing_df
        new_df.reset_index(inplace=True)
        if user_setter:
            self.df = new_df  # use setter for validation and hooks
        else:
            self._df = self._prep_input_df(new_df)

    @property
    def number(self) -> int:
        """Return the number of entries in the DataFrame df."""
        return len(self._df) if self._df is not None else 0

    def _check_frozen(self):
        if self._frozen:
            raise AttributeError(f"This {self.__class__.__name__} instance is frozen and cannot be modified.")


class LogbooksSchema(pa.DataFrameModel):
    """Pandera schema for Logbooks Dataframe validation."""

    id_journey: int = pa.Field(ge=1, coerce=True)
    id_vehicle: int = pa.Field(ge=1, coerce=True)
    dep_dt: pa.DateTime = pa.Field(coerce=True)
    arr_dt: pa.DateTime = pa.Field(coerce=True)
    dep_loc: int = pa.Field(ge=1, coerce=True)
    arr_loc: int = pa.Field(ge=1, coerce=True)
    distance: float = pa.Field(gt=0)

    class Config:
        strict = "filter"  # remove extra columns
        coerce = True  # enforce dtypes
        ordered = False  # don't enforce column order

    # check that dep_dt is before arr_dt
    @pa.dataframe_check(
        error="Departure time (dep_dt) must be before arrival time (arr_dt) for all journeys.", groupby=None
    )
    def check_time_order(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure dep_dt is before arr_dt for all journeys."""
        return df["dep_dt"] < df["arr_dt"]

    # check no overlapping journeys per vehicle
    @pa.dataframe_check(
        error="Journeys for the same vehicle cannot overlap. dep_dt must be >= previous arr_dt.", groupby=None
    )
    def check_no_overlapping_journeys(cls, df: pd.DataFrame) -> Series[bool]:
        """Check for no overlapping journeys per vehicle."""
        # Get previous arr_dt per vehicle
        prev_arr_dt = df.groupby("id_vehicle")["arr_dt"].shift(1)

        # First journey per vehicle is always valid
        # For others, dep_dt must be after previous arr_dt
        is_first_journey = prev_arr_dt.isna()
        no_overlap = df["dep_dt"] >= prev_arr_dt

        return is_first_journey | no_overlap


class Logbooks(BaseMobilityComponent):
    """
    Component class included in :class:`MobProfiles` representing the logbooks with all journeys.

    The Logbooks class represents the logbook data of journeys, including departure and arrival times, locations, and distances.
    The class holding a dataframe df that contains the data.
    It is included as a component in the :class:`MobProfiles` class and can be accessed via its instances.
    It provides methods to add, update, and delete journeys, as well as to restore location continuity and convert temporal resolution. 
    The Logbooks class ensures data integrity through validation with a Pandera schema.

    The DataFrame (accessible via :attr:`~champpy.Logbooks.df`) contains the following columns:

    .. list-table::
       :header-rows: 1

       * - Column
         - Type
         - Description
       * - id_journey
         - :class:`int`
         - One-based index for journeys. This column is optional will be generated if not provided in the input DataFrame.
       * - id_vehicle
         - :class:`int`
         - One-based index for vehicles, connected to id_vehicle in input_vehicles_df.
       * - dep_dt
         - :class:`pandas.Timestamp`
         - Departure datetime of each journey.
       * - arr_dt
         - :class:`pandas.Timestamp`
         - Arrival datetime of each journey.
       * - dep_loc
         - :class:`int`
         - Departure location of each journey as integer above 0.
           You can for example define 1 for home, 2 for work, etc.
           The location = 0 is reserved for driving and not allowed in this dataframe.
       * - arr_loc
         - :class:`int`
         - Arrival location of each journey as integer above 0.
           You can for example define 1 for home, 2 for work, etc.
           The location = 0 is reserved for driving and not allowed in this dataframe.
       * - distance
         - :class:`float`
         - Distance of each journey in km.
       * - duration
         - :class:`float`
         - Duration of each journey in hours.
       * - speed
         - :class:`float`
         - Speed of each journey in km/h.

    Parameters
    ----------
    input_df : :class:`pandas.DataFrame`
        Input DataFrame for the logbooks. Please see column description in :class:`Logbooks` for required columns and types.
        The column `id_journey` is optional and will be generated if not provided in the input DataFrame.
        The columns `duration` and `speed` are not required as they are calculated. They will be ignored if provided in the input DataFrame.
                
    frozen : bool, optional
        If True, the Logbooks instance is immutable after creation. Default is False.   
    """

    _schema = LogbooksSchema  # Pandera schema for validation of the logbooks DataFrame

    def __init__(self, input_df: pd.DataFrame = None, frozen: bool = False):
        """
        Initialize a Logbooks object.

        The parameters are described in the class docstring. 
        """
        self._event_on_locations = Event[self]()  # Event triggered on logbooks update
        super().__init__(input_df=input_df, frozen=frozen)  # call base constructor
        self._temp_res = None  # temporal resolution in hours

    @staticmethod
    def _prep_input_df(input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare input DataFrame for the logbook.:
        - Sort Dataframe by id_vehicle and dep_dt
        - Add id_journey to Dataframe if missing
        - Validate Dataframe using Pandera schema
        - Sort columns to standard order
        """
        # Return empty df if input is empty
        if input_df is None or input_df.empty:
            return

        # Sorted required logbook rows based on id_vehicle and dep_dt
        if {"id_vehicle", "dep_dt"}.issubset(input_df.columns):
            input_df = input_df.sort_values(by=["id_vehicle", "dep_dt"]).reset_index(drop=True)

        # Add id_journey if missing
        if "id_journey" not in input_df.columns:
            input_df.insert(0, "id_journey", range(1, len(input_df) + 1))

        # Validate using Pandera schema
        LogbooksSchema.validate(input_df)

        # Sort columns to standard order
        required_cols = ["id_journey", "id_vehicle", "dep_dt", "arr_dt", "dep_loc", "arr_loc", "distance"]
        input_df = input_df[required_cols]

        return input_df

    def _on_df_getter(self, output_df) -> pd.DataFrame:
        """Add duration and speed columns to output_df for the getter."""
        duration = (self._df["arr_dt"] - self._df["dep_dt"]).dt.total_seconds() / 3600  # in hours
        speed = self._df["distance"] / duration  # in km/h
        return output_df.assign(duration=duration, speed=speed)

    def _on_df_setter(self):
        """Call restore_location_continuity after setting new dataframe."""
        self._df = self._df.sort_values(by=["id_vehicle", "dep_dt"]).reset_index(drop=True)
        self.restore_location_continuity()
        # Triggger event to update location labels
        self._event_on_locations.trigger(self)

    def add_journeys(self, input_df: pd.DataFrame) -> None:
        """
        Add journeys from a DataFrame to the logbook.

        Parameters
        ----------
        input_df : pandas.DataFrame
            DataFrame with journey data. Please see column description in :class:`Logbooks` for required columns and types.
            The columns `duration` and `speed` are not required as they are calculated. They will be ignored if provided in the input DataFrame.

        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Create new journeys DataFrame
            new_journeys_df = pd.DataFrame({
                "id_vehicle": [1, 1],
                "dep_dt": [pd.Timestamp("2024-01-01 08:00"), pd.Timestamp("2024-01-01 10:00")],
                "arr_dt": [pd.Timestamp("2024-01-01 09:00"), pd.Timestamp("2024-01-01 11:00")],
                "dep_loc": [1, 2],
                "arr_loc": [2, 3],
                "distance": [10.0, 15.0]
            })

            # Add journeys to logbooks
            mob_profiles.logbooks.add_journeys(new_journeys_df)

        """
        # Prepare input DataFrame
        prepared_df = self._prep_input_df(input_df)

        # Generate id_journey for new journeys
        prepared_df["id_journey"] = prepared_df["id_journey"] + self.number

        # copy of existiing df
        existing_df = self.df

        # Append to existing DataFrame
        existing_df = pd.concat([existing_df, prepared_df], ignore_index=True)

        # Sort by id_vehicle and dep_dt
        existing_df = existing_df.sort_values(by=["id_vehicle", "dep_dt"]).reset_index(drop=True)

        # use setter for validation and hooks
        self.df = existing_df

    def update_journeys(self, input_df: pd.DataFrame) -> None:
        """
        Update existing journeys in the logbook based on id_journey.

        Parameters
        ----------
        input_df : pandas.DataFrame
                DataFrame with journey data. Please see column description in :class:`Logbooks` for required columns and types.
                Must include `id_journey` column.
                The columns `duration` and `speed` are not required as they are calculated. They will be ignored if provided in the input DataFrame.

        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Get the data of the first two journeys and modify its departure times and distance
            updated_journeys_df = mob_profiles.logbooks.df.head(2)

            updated_journeys_df.loc[:, "arr_dt"] = updated_journeys_df.loc[:, "arr_dt"] + pd.Timedelta(minutes=30)
            updated_journeys_df.loc[:, "distance"] = updated_journeys_df.loc[:, "distance"] + 5.0

            # Update journeys in logbooks
            mob_profiles.logbooks.update_journeys(updated_journeys_df)

        """
        # Update journeys using base class method
        self._update_rows_of_df(input_df, index_cols=["id_journey"], user_setter=True, prefer_input=False)

    @validate_call
    def delete_journeys(self, id_journey: list) -> None:
        """Delete journeys by journey ID.

        Parameters
        ----------
        id_journey : list[int]
                List of journey IDs to delete.

        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Delete the first two journeys of the logbook
            mob_profiles.logbooks.delete_journeys(id_journey=[1, 2])

        """
        # Build deletion mask and deltete rows
        mask_delete = self._df["id_journey"].isin(id_journey)
        self._del_rows_of_df(mask_delete)

        # Restore location continuity after deletion
        self.restore_location_continuity()

    def _delete_vehicles(self, id_vehicle: list) -> None:
        """Delete all journeys of specific vehicles.

        Parameters
        ----------
        id : list[int]
                List of vehicle IDs whose journeys should be deleted.
        """
        # Build deletion mask and deltete rows
        mask_delete = self._df["id_vehicle"].isin(id_vehicle)
        self._del_rows_of_df(mask_delete)

    @validate_call
    def restore_location_continuity(self, target: Literal["dep", "arr"] = "dep") -> None:
        """
        Restore location continuity by overwriting either dep_loc or arr_loc.

        Meaning location continuity: the departure location (dep_loc) of every journey for a vehicle
        must have the same value as the arrival location (arr_loc) of the previous journey.

        Parameters
        ----------
        target : :class:`Literal`["dep", "arr"], optional
            "dep" (default): set dep_loc to previous arr_loc.
            "arr": set arr_loc to next dep_loc.
        """
        if self._df is None or self._df.empty:
            return

        # Ensure ordering per vehicle
        self._df = self._df.sort_values(["id_vehicle", "dep_dt"]).reset_index(drop=True)

        if target == "dep":
            prev_arr_loc = self._df.groupby("id_vehicle")["arr_loc"].shift(1)
            mask = ~prev_arr_loc.isna() & (self._df["dep_loc"] != prev_arr_loc)
            self._df.loc[mask, "dep_loc"] = prev_arr_loc[mask]
        elif target == "arr":
            next_dep_loc = self._df.groupby("id_vehicle")["dep_loc"].shift(-1)
            mask = ~next_dep_loc.isna() & (self._df["arr_loc"] != next_dep_loc)
            self._df.loc[mask, "arr_loc"] = next_dep_loc[mask]
        else:
            message = "target must be either 'dep' or 'arr'"
            logger.error(message)
            raise ValueError(message)

    @property
    def temp_res(self) -> float:
        """
        Temporal resolution of the logbook in hours.

        :getter: Returns the current temporal resolution of the logbook in hours. 
                 If no temporal resolution has been set, returns None.
        :setter: Set the temporal resolution of the logbook in hours. 
                 This will convert the logbook to the specified temporal resolution 
                 by merging overlapping/adjacent journeys per vehicle.
        
        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Get current temporal resolution (initially None)
            current_res = mob_profiles.logbooks.temp_res
            
            # Set temporal resolution to 1 hour
            # This will merge journeys that overlap or are adjacent within 1-hour intervals
            mob_profiles.logbooks.temp_res = 1.0
            
            # Check the new temporal resolution
            print(mob_profiles.logbooks.temp_res)  # Output: 1.0
        """
        return self._temp_res

    @temp_res.setter
    def temp_res(self, value: float):
        """Set the temporal resolution of the logbook in hours."""
        if value <= 0:
            message = "Temporal resolution must be a positive number."
            logger.error(message)
            raise ValueError(message)
        self._convert_temporal_resolution(value)
        self._temp_res = value

    def _convert_temporal_resolution(self, temp_res: float) -> None:
        """
        Convert the logbook to a specified temporal resolution (in hours),
        merging overlapping/adjacent journeys per vehicle.

        This temporal resolution is applied as follows:
        - Round `dep_dt` down to the resolution grid (floor)
        - Round `arr_dt` up to the resolution grid (ceil)
        - Within each vehicle, merge consecutive journeys whose rounded
          dep_dt <= previous rounded arr_dt OR whose rounded arr_dt equals previous arr_dt
        - Aggregate per merged block: first dep_dt/dep_loc, last arr_dt/arr_loc, sum distance
        - Return aggregated logbook (schema columns)

        Parameters
        ----------
        temp_res : float
                Temporal resolution in hours.

        Returns
        -------
        pd.DataFrame
                Aggregated logbook DataFrame (columns: id_journey, id_vehicle, dep_dt, arr_dt, dep_loc, arr_loc, distance).
        """
        # Empty guard
        if self._df is None or self._df.empty:
            return LogbooksSchema.example(size=0)

        # create copy of dataframe
        df = self.df

        # Round departure down (floor) and arrival up (ceil) to resolution grid
        freq = f"{int(temp_res * 60)}min"
        dep_floor = df["dep_dt"].dt.floor(freq)
        arr_ceil = df["arr_dt"].dt.ceil(freq)

        # If already aligned to resolution, return original (schema columns)
        if df["dep_dt"].equals(dep_floor) and df["arr_dt"].equals(arr_ceil):
            return df[["id_journey", "id_vehicle", "dep_dt", "arr_dt", "dep_loc", "arr_loc", "distance"]]

        # Prepare rounded dataframe
        df["dep_dt_r"] = dep_floor
        df["arr_dt_r"] = arr_ceil

        # Ensure ordering per vehicle by rounded dep_dt
        df = df.sort_values(["id_vehicle", "dep_dt_r"]).reset_index(drop=True)

        # Determine group boundaries per vehicle
        prev_arr = df.groupby("id_vehicle")["arr_dt_r"].shift(1)
        same_group = (df["dep_dt_r"] <= prev_arr) | (df["arr_dt_r"] == prev_arr)
        new_group_flag = (~same_group) | prev_arr.isna()
        df["grp_idx"] = new_group_flag.groupby(df["id_vehicle"]).cumsum()

        # Aggregate per (id_vehicle, grp_idx)
        grouped = df.groupby(["id_vehicle", "grp_idx"], sort=False)
        agg_df = grouped.agg(
            id_vehicle=("id_vehicle", "first"),
            dep_dt=("dep_dt_r", "first"),
            arr_dt=("arr_dt_r", "last"),
            dep_loc=("dep_loc", "first"),
            arr_loc=("arr_loc", "last"),
            distance=("distance", "sum"),
        ).reset_index(drop=True)

        # set agregated df as logbook df using setter for validation and hooks
        self.df = agg_df


class VehiclesSchema(pa.DataFrameModel):
    """Pandera schema for Vehicles Dataframe validation."""

    id_vehicle: int = pa.Field(ge=1, coerce=True)
    first_day: pa.DateTime = pa.Field(coerce=True)
    last_day: pa.DateTime = pa.Field(coerce=True)
    id_cluster: int = pa.Field(ge=1, coerce=True, default=1)
    first_loc: Series[pd.Int64Dtype] = pa.Field(ge=0, nullable=True, coerce=True, default=None)

    class Config:
        strict = "filter"  # remove extra columns
        coerce = True  # enforce dtypes
        ordered = False  # don't enforce column order
        add_missing_columns = True

    # check that dep_dt is before arr_dt
    @pa.dataframe_check(
        error="First day (first_day) must be before last day (last_day) for all vehicles.", groupby=None
    )
    def check_time_order(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure first_day is before last_day for all vehicles."""
        return df["first_day"] <= df["last_day"]

    # check id_vehicle is unique
    @pa.dataframe_check(error="id_vehicle must be unique. No duplicate vehicle IDs allowed.", groupby=None)
    def check_id_vehicle_unique(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure id_vehicle is unique across all rows."""
        return ~df["id_vehicle"].duplicated(keep=False)


class Vehicles(BaseMobilityComponent):
    """
    Component class included in :class:`MobProfiles` representing vehicles.

    The Vehicles class manages vehicle-level metadata.
    It is included as a component in the :class:`MobProfiles` class and can be accessed via its instances.

    The DataFrame (accessible via :attr:`~champpy.Vehicles.df`) contains the following columns:

    .. list-table::
        :header-rows: 1

        * - Column
          - Type
          - Description
        * - id_vehicle
          - :class:`int`
          - Vehicle identifier. One-based index for vehicles.
        * - first_day
          - :class:`pandas.Timestamp`
          - First recorded day of the vehicle.
        * - last_day
          - :class:`pandas.Timestamp`
          - Last recorded day of the vehicle.
        * - id_cluster
          - :class:`int`
          - Cluster assignment (optional, default: 1). 
            Used to group vehicles into different clusters.
        * - first_loc
          - :class:`int`
          - First location of the vehicle (optional, default: None). 
            Use the same location encoding as in the logbooks.
    
    Parameters
    ----------
    input_df : :class:`pandas.DataFrame`
        Input DataFrame for the vehicles. Please see column description above for required columns and types.

    frozen : bool, optional
        If True, the Vehicles instance is immutable after creation. Default is False.
    """

    _schema = VehiclesSchema  # Pandera schema for validation of the vehicles DataFrame

    def __init__(self, input_df: pd.DataFrame = None, frozen: bool = False):
        """
        Initialize a Vehiclesobject.

        Parameters
        ----------
        input_df : pd.DataFrame, optional
                Initial DataFrame with vehicle data. See column description above.

        frozen : bool, optional
                If True, the Vehicles instance is immutable after creation. Default is False.
        """
        self._event_on_logbooks = Event[int]()  # Event triggered on vehicle deletion
        self._event_on_clusters = Event[self]()  # Event triggered on vehicle update
        super().__init__(input_df=input_df, frozen=frozen)  # call base constructor

    def _on_df_setter(self):
        """Call restore_location_continuity after setting new dataframe."""
        # Triggger event to update cluster labels
        self._event_on_clusters.trigger(self)

    def add_vehicles(self, input_df: pd.DataFrame) -> None:
        """
        Add vehicles from a DataFrame.

        Parameters
        ----------  
        input_df : pd.DataFrame
                DataFrame with vehicle data to add. See column description table in :class:`Vehicles` for required columns.

        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Create new vehicles DataFrame
            new_vehicles_df = pd.DataFrame({
                "id_vehicle": [3, 4],
                "first_day": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "last_day": pd.to_datetime(["2020-01-03", "2020-01-04"]),
                "id_cluster": [1, 1],
                "first_loc": [1, 2]
            })
            # Add vehicles from a DataFrame
            mob_profiles.vehicles.add_vehicles(input_df=new_vehicles_df)
        """
        # Validate input DataFrame
        new_vehicles_df = VehiclesSchema.validate(input_df)

        # Create copy of existing df
        existing_df = self.df

        # Append to existing DataFrame
        new_df = pd.concat([existing_df, new_vehicles_df], ignore_index=True)

        # use setter for validation and hooks
        self.df = new_df

    def update_vehicles(self, input_df: pd.DataFrame) -> None:
        """
        Update existing vehicles based on id_vehicle. Replaces all columns for matching vehicles with values from input_df.

        Parameters
        ----------
        input_df : pd.DataFrame
                DataFrame with vehicle data to add. See column description table in :class:`Vehicles` for required columns.
        
        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Get dataframe of the second vehicle and set its cluster to 2
            updated_vehicles_df = mob_profiles.vehicles.df[mob_profiles.vehicles.df["id_vehicle"] == 2]
            updated_vehicles_df.loc[:, "id_cluster"] = 2
            
            # Update vehicles from a DataFrame
            mob_profiles.vehicles.update_vehicles(input_df=updated_vehicles_df)
        """
        # Update vehicles using base class method
        self._update_rows_of_df(input_df, index_cols=["id_vehicle"], user_setter=True, prefer_input=False)

    def delete_vehicles(self, id_vehicle: list) -> None:
        """Delete vehicles by vehicle ID.

        Parameters
        ----------
        id_vehicle : list[int]
                List of vehicle IDs to delete.
        
        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Delete the second vehicle and all its journeys
            mob_profiles.vehicles.delete_vehicles(id_vehicle=[2])
        """
        # Build deletion mask
        mask = self._df["id_vehicle"].isin(id_vehicle)
        self._del_rows_of_df(mask)

        # Triggger event to update cluster labels and logbooks
        self._event_on_logbooks.trigger(id_vehicle)
        self._event_on_clusters.trigger(self)

    def generate_vehicles_from_logbooks(self, logbooks: Logbooks) -> None:
        """
        Generate vehicle DataFrame from a Logbooks instance.

        Parameters
        ----------
        logbooks : :class:`Logbooks`
                Logbooks instance with journey data to generate vehicles from.
        """
        if isinstance(logbooks, Logbooks) == False:
            message = "logbooks must be an instance of Logbooks class."
            logger.error(message)
            raise TypeError(message)

        logbooks_df = logbooks.df

        if logbooks_df is None or logbooks_df.empty:
            self._df = VehiclesSchema.example(size=0)
            return

        # Group by id_vehicle to get first_day, last_day and first_location
        grouped = (
            logbooks_df.groupby("id_vehicle")
            .agg(
                first_day=pd.NamedAgg(column="dep_dt", aggfunc="min"),
                last_day=pd.NamedAgg(column="arr_dt", aggfunc="max"),
                first_loc=pd.NamedAgg(column="dep_loc", aggfunc="first"),
            )
            .reset_index()
        )

        # Convert to date only
        grouped["first_day"] = grouped["first_day"].dt.floor("D")
        grouped["last_day"] = grouped["last_day"].dt.floor("D")

        # Assign cluster as 1 for all vehicles (placeholder)
        grouped["cluster"] = 1

        # Save as vehicles DataFrame using setter for validation and hooks
        self.df = grouped

    def set_first_loc_from_logbooks(self, logbooks: Logbooks) -> None:
        """
        Set first_loc for each vehicle based on the first dep_loc in the logbooks.

        Parameters
        ----------
        logbooks : :class:`Logbooks`
                Logbook instance with journey data to extract first locations from.
        """
        if isinstance(logbooks, Logbooks) == False:
            message = "logbooks must be an instance of Logbook class."
            logger.error(message)
            raise TypeError(message)

        logbooks_df = logbooks.df

        if logbooks_df is None or logbooks_df.empty:
            return

        # Get first dep_loc per vehicle
        first_loc = logbooks_df.sort_values(by=["dep_dt"]).groupby("id_vehicle").first().reset_index()
        first_loc = first_loc[["id_vehicle", "dep_loc"]].rename(columns={"dep_loc": "first_loc"})

        # Remove existing first_loc column if present to avoid _x/_y suffix
        if "first_loc" in self._df.columns:
            self._df = self._df.drop(columns=["first_loc"])

        # Create a copy of the vehicle DataFrame
        existing_df = self.df

        # Merge into vehicle DataFrame
        new_df = existing_df.merge(first_loc, on="id_vehicle", how="left")

        # set first_loc of non driving vehicles to 1: nan --> 1
        new_df.loc[new_df["first_loc"].isna(), "first_loc"] = 1
        new_df["first_loc"] = new_df["first_loc"].astype("Int64")

        # Use setter for validation and hooks
        self.df = new_df


class ClustersSchema(pa.DataFrameModel):
    """Pandera schema for Logbooks Dataframe validation."""

    id_cluster: int = pa.Field(ge=1, coerce=True)
    label: str = pa.Field(coerce=True)


class Clusters(BaseMobilityComponent):
    """
    Component class included in :class:`MobProfiles` representing vehicle clusters.

    The Clusters class manages cluster assignments for vehicles in the mobility data.
    It is included as a component in the :class:`MobProfiles` class and can be accessed via its instances.
    The clusters DataFrame is automatically generated from the vehicles DataFrame 
    and cannot be set directly, but can be updated via the update methods.

    The DataFrame (accessible via :attr:`~champpy.Clusters.df`) contains the following columns:

    .. list-table::
        :header-rows: 1

        * - Column
          - Type
          - Description
        * - id_cluster
          - :class:`int`
          - Cluster identifier.
        * - label
          - :class:`str`
          - Human-readable label for the cluster.

    Parameters
    ----------
    vehicles : :class:`Vehicles`, optional
        Vehicles instance with vehicle data including 'id_cluster' column.
        If provided, clusters will be automatically generated from the unique cluster IDs.
    frozen : bool, optional
        If True, the Clusters instance is immutable after creation. Default is False.
    """

    _schema = ClustersSchema  # Pandera schema for validation of the clusters DataFrame

    def __init__(self, vehicles: Vehicles | None = None, frozen: bool = False):
        """
        Initialize a Clusters object.

        The parameters are described in the class docstring.
        """
        super().__init__(input_df=None)  # call base constructor
        if vehicles is None:
            # Initialize empty clusters DataFrame
            self._df = ClustersSchema.example(size=0)
        elif isinstance(vehicles, Vehicles):
            # Generate clusters from vehicles
            self._df = pd.DataFrame()
            self.update_clusters_from_vehicles(vehicles)
        else:
            message = "vehicles must be an instance of Vehicles class."
            logger.error(message)
            raise TypeError(message)
        self._frozen = frozen

    @BaseMobilityComponent.df.setter
    def df(self, value: pd.DataFrame):
        """Not allowed to set clusters DataFrame directly."""
        mssg = "Setting clusters DataFrame directly is not allowed. Use update methods instead: update_clusters_from_vehicles(), update_clusters()."
        logger.error(mssg)
        raise AttributeError(mssg)

    def update_clusters_from_vehicles(self, vehicles: Vehicles) -> None:
        """
        Update clusters DataFrame based on current vehicle DataFrame.

        Parameters
        ----------
        vehicles : Vehicles
                Vehicles instance with vehicle data including 'id_cluster' column.
        """
        # Get copy of vehicles DataFrame
        vehicles_df = vehicles.df

        # Create clusters DataFrame from unique id_cluster in vehicles
        cluster_ids = vehicles_df["id_cluster"].unique()
        cluster_labels = [f"Cluster {cid}" for cid in cluster_ids]
        update_df = pd.DataFrame({"id_cluster": cluster_ids, "label": cluster_labels})

        # Update clusters DataFrame using function of base class
        self._update_rows_of_df(update_df, index_cols=["id_cluster"], user_setter=False, prefer_input=True)

    def update_clusters(self, input_df: pd.DataFrame) -> None:
        """
        Update existing clusters based on id_cluster. Replaces all columns for matching clusters with values from input_df.

        Parameters
        ----------
        input_df : pd.DataFrame
                DataFrame with cluster data to update. 
                See column description table in :class:`Clusters` for required columns.

        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Get current clusters DataFrame
            clusters_df = mob_profiles.clusters.df
            
            # Update cluster labels
            clusters_df.loc[clusters_df["id_cluster"] == 1, "label"] = "Private Vehicles"
            
            # Apply updated labels
            mob_profiles.clusters.update_clusters(clusters_df)
        """
        # Update clusters DataFrame using function of base class
        self._update_rows_of_df(input_df, index_cols=["id_cluster"], user_setter=False, prefer_input=False)


class LocationsSchema(pa.DataFrameModel):
    """Pandera schema for Logbooks Dataframe validation."""

    location: int = pa.Field(ge=0, coerce=True)
    label: str = pa.Field(coerce=True)


class Locations(BaseMobilityComponent):
    """
    Component class included in :class:`MobProfiles` representing locations used in journeys.

    The Locations class manages location definitions for the mobility data.
    It is included as a component in the :class:`MobProfiles` class and can be accessed via its instances.
    The locations DataFrame is automatically generated from the logbooks and vehicles DataFrames
    and cannot be set directly, but can be updated via the update methods.
    Location 0 is reserved for "Driving" and location 1 is typically "Home".

    The DataFrame (accessible via :attr:`~champpy.Locations.df`) contains the following columns:

    .. list-table::
        :header-rows: 1

        * - Column
          - Type
          - Description
        * - location
          - :class:`int`
          - Location identifier (0 = Driving, 1+ = stationary locations).
        * - label
          - :class:`str`
          - Human-readable label for the location (e.g., "Home", "Work", "Location 3").

    Parameters
    ----------
    vehicles : :class:`Vehicles`, optional
        Vehicles instance to extract first_loc values from.
    logbooks : :class:`Logbooks`, optional
        Logbooks instance to extract dep_loc and arr_loc values from.
    frozen : bool, optional
        If True, the Locations instance is immutable after creation. Default is False.
    

    """

    _schema = LocationsSchema  # Pandera schema for validation of the locations DataFrame

    def __init__(self, vehicles: Vehicles | None = None, logbooks: Logbooks | None = None, frozen: bool = False):
        """
        Initialize a Locations object.

        The parameters are described in the class docstring.
        """
        super().__init__(input_df=None)  # call base constructor
        self.update_locations_from_logbooks_vehicles(logbooks=logbooks, vehicles=vehicles)
        self._frozen = frozen

    @BaseMobilityComponent.df.setter
    def df(self, value: pd.DataFrame):
        """Not allowed to set locations DataFrame directly."""
        mssg = "Setting locations DataFrame directly is not allowed. Use update methods instead: update_locations_from_logbooks_vehicles()."
        logger.error(mssg)
        raise AttributeError(mssg)

    def update_locations_from_logbooks_vehicles(
        self, logbooks: Optional[Logbooks] = None, vehicles: Optional[Vehicles] = None
    ) -> None:
        """
        Update locations DataFrame based on unique dep_loc and arr_loc in logbooks.

        Parameters
        ----------
        logbooks : Optional[Logbooks]
                Logbooks instance with journey data to extract locations from.
        vehicles : Optional[Vehicles]
                Vehicles instance with vehicle data to extract locations from.
        """
        if vehicles is None and logbooks is None:
            message = "At least one of vehicles or logbooks must be provided."
            logger.error(message)
            raise ValueError(message)
        if vehicles is not None and not isinstance(vehicles, Vehicles):
            message = "vehicles must be an instance of Vehicles class."
            logger.error(message)
            raise TypeError(message)
        if logbooks is not None and not isinstance(logbooks, Logbooks):
            message = "logbooks must be an instance of Logbooks class."
            logger.error(message)
            raise TypeError(message)

        logbooks_df = logbooks.df

        if logbooks_df is None or logbooks_df.empty:
            return

        # Get unique locations from vehicles and logbooks
        all_locs = [0]  # include location 0 by default for driving
        if vehicles is not None:
            loc_vehicles = vehicles.df["first_loc"].dropna().unique().tolist()
            all_locs.extend(loc_vehicles)
        if logbooks is not None:
            dep_locs = logbooks_df["dep_loc"].unique().tolist()
            arr_locs = logbooks_df["arr_loc"].unique().tolist()
            all_locs.extend(dep_locs)
            all_locs.extend(arr_locs)
        all_locs = sorted(set(all_locs))

        # Create new locations DataFrame
        new_locations_df = pd.DataFrame({"location": all_locs, "label": [f"Location {loc}" for loc in all_locs]})
        # Update locations DataFrame: 0 = driving, 1 = home
        new_locations_df.loc[new_locations_df["location"] == 0, "label"] = "Driving"
        new_locations_df.loc[new_locations_df["location"] == 1, "label"] = "Home"

        self._update_rows_of_df(new_locations_df, index_cols=["location"], user_setter=False, prefer_input=True)

    def update_locations(self, input_df: pd.DataFrame) -> None:
        """
        Update existing locations based on location ID. Replaces all columns for matching locations with values from input_df.

        Parameters
        ----------
        input_df : pd.DataFrame
                DataFrame with location data to update. See column description table in :class:`Locations` for required columns.

        Examples
        --------
        This example uses the instance `mob_profiles` defined in the :class:`MobProfiles` examples:

        .. code-block:: python

            # Get current locations DataFrame
            locations_df = mob_profiles.locations.df
            
            # Update location labels with meaningful names
            locations_df.loc[locations_df["location"] == 2, "label"] = "Work"
            
            # Apply updated labels
            mob_profiles.locations.update_locations(locations_df)
        """
        self._update_rows_of_df(input_df, index_cols=["location"], user_setter=False, prefer_input=False)
