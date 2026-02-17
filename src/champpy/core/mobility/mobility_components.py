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

    def __init__(self, input_df: pd.DataFrame | None = None, frozen: bool = False):
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
        """Get the DataFrame of the mobility component."""
        if self._df is None:
            output_df = self._schema.example(size=0)
        else:
            output_df = self._df.copy()
        output_df = self._on_df_getter(output_df)  # Hook method for subclasses
        return output_df

    @df.setter
    def df(self, input_df: pd.DataFrame):
        """Set the DataFrame of the mobility component with validation."""
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
        """Return the number of entries in the DataFrame."""
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
    Class representing a logbook for vehicle journeys.
    """

    _schema = LogbooksSchema  # Pandera schema for validation of the logbooks DataFrame

    def __init__(self, input_df: pd.DataFrame = None, frozen: bool = False):
        """
        Initialize a Logbooks object.

        Parameters
        ----------
        input_df : pd.DataFrame, optional
                Initial DataFrame with journey data.
                Expected columns and dtypes:
                - id_journey: int
                - id_vehicle: int
                - dep_dt: datetime64[ns]
                - arr_dt: datetime64[ns]
                - dep_loc: str
                - arr_loc: str
                - distance: float
        frozen : bool, optional
                If True, the Logbooks instance is immutable after creation. Default is False.
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
        input_df : pd.DataFrame
                DataFrame with journey data to add including columns:
                - id_vehicle: int
                - dep_dt: datetime64[ns]
                - arr_dt: datetime64[ns]
                - dep_loc: str
                - arr_loc: str
                - distance: float
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
        input_df : pd.DataFrame
                DataFrame with journey data to update.
                Must include 'id_journey' column.
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
        target : Literal["dep", "arr"], optional
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
        """Get the temporal resolution of the logbook in hours."""
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
    Class representing vehicles with aggregated statistics from journeys.
    """

    _schema = VehiclesSchema  # Pandera schema for validation of the vehicles DataFrame

    def __init__(self, input_df: pd.DataFrame = None, frozen: bool = False):
        """
        Initialize a Vehiclesobject.

        Parameters
        ----------
        input_df : pd.DataFrame, optional
                Initial DataFrame with vehicle data.
                Expected columns and dtypes:
                - id_vehicle: int
                - first_day: datetime64[D]
                - last_day:  datetime64[D]
                - id_cluster:   int (optional: default 1)
                - first_loc: int (optional: Default None)
        frozen : bool, optional
                If True, the Vehiclesinstance is immutable after creation. Default is False.
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
                DataFrame with vehicle data to add.
                Must include the following columns:
                - id_vehicle: int
                - first_day: datetime64[D]
                - last_day:  datetime64[D]
                - id_cluster:   int
                - first_loc: int (optional)
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
                DataFrame with vehicle data to update.
                Must include the following columns:
                - id_vehicle: int
                - first_day: datetime64[D]
                - last_day:  datetime64[D]
                - id_cluster:   int
                - first_loc: int (optional)
        """
        # Update vehicles using base class method
        self._update_rows_of_df(input_df, index_cols=["id_vehicle"], user_setter=True, prefer_input=False)

    def delete_vehicles(self, id_vehicle: list) -> None:
        """Delete vehicles by vehicle ID.

        Parameters
        ----------
        id_vehicle : list[int]
                List of vehicle IDs to delete.
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
        logbooks : Logbooks
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
        logbooks : Logbooks
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
    "Class representing clusters of vehicles."

    _schema = ClustersSchema  # Pandera schema for validation of the clusters DataFrame

    def __init__(self, vehicles: Vehicles | None = None, frozen: bool = False):
        """
        Generate clusters DataFrame.

        Parameters
        ----------
        vehicles : Vehicles, optional
                Vehicles instance with vehicle data including 'id_cluster' column.
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
                Must include the following columns:
                - id_cluster: int
                - label: str
        """
        # Update clusters DataFrame using function of base class
        self._update_rows_of_df(input_df, index_cols=["id_cluster"], user_setter=False, prefer_input=False)


class LocationsSchema(pa.DataFrameModel):
    """Pandera schema for Logbooks Dataframe validation."""

    location: int = pa.Field(ge=0, coerce=True)
    label: str = pa.Field(coerce=True)


class Locations(BaseMobilityComponent):
    "Class representing locations used in journeys."

    _schema = LocationsSchema  # Pandera schema for validation of the locations DataFrame

    def __init__(self, vehicles: Vehicles | None = None, logbooks: Logbooks | None = None, frozen: bool = False):
        """
        Initialize Locations instance.

        Parameters
        ----------
        input_df : pd.DataFrame, optional
                Initial DataFrame with location data.
                Expected columns and dtypes:
                - location: int
                - label: str
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
                DataFrame with location data to update.
                Must include the following columns:
                - location: int
                - label: str
        """
        self._update_rows_of_df(input_df, index_cols=["location"], user_setter=False, prefer_input=False)
