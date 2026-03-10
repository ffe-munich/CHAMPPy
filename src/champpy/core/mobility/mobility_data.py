import pandas as pd
import numpy as np
from typing import Literal
import logging

from pydantic import validate_call
from champpy.core.mobility.mobility_components import (
    Logbooks,
    Vehicles,
    Clusters,
    Locations,
)
from champpy.utils.time_utils import get_datetime_array

logger = logging.getLogger(__name__)


class MobProfiles:
    """
    Wrapper class for mobility profiles in the champpy framework.

    It contains the logbooks, vehicles, clusters and locations as separate classes.

    Parameters
    ----------
    input_logbooks_df : :class:`pandas.DataFrame`
        Input DataFrame for the logbooks.
        Expected columns and dtypes:

        .. list-table::
           :header-rows: 1

           * - Column
             - Type
             - Description
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

    input_vehicles_df : :class:`pandas.DataFrame`, optional
        Input DataFrame for the vehicles. If not provided, the vehicles will be generated from the logbooks.
        Expected columns and dtypes:

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
           * - cluster
             - :class:`int`
             - Split the vehicles into clusters by assigning a cluster ID (one-based) to each vehicle.
               This is optional and can be used for example to distinguish between different user groups.
               If you don't want to use clusters, you can simply set the cluster column to 1 for all vehicles.
           * - first_loc
             - :class:`int`
             - First location (optional). Use the same location encoding as in dep_loc and arr_loc in input_logbooks_df.
               It is espacially relevant for non-driving vehicles, which do not have any journeys in the logbooks.

    frozen : :class:`bool`, optional
        If True, the MobProfiles instance is immutable after creation. Default is False.

    Attributes
    ----------
    logbooks : :class:`Logbooks`
            Contains the journey data of the mobility profile with departure and arrival information.
    vehicles : :class:`Vehicles`
            Contains vehicle-specific data about eaach vehicle,
            such as its first and last day of activity, cluster assignment, and first location.
            It is connected to logbooks via id_vehicle.
    clusters : :class:`Clusters`
            Describes the clusters defined in vehicles. It is connected to vehicles via id_cluster.
            It provides a label for each cluster.
    locations : :class:`Locations`
            Describes the locations defined in logbooks and vehicles. The location is connected to logbooks via dep_loc and arr_loc and to vehicles via first_loc.
            It provides a label for each location. The location = 0 is reserved for driving and gets the label "Driving".

    Examples
    --------
    Create a MobProfiles instance with minimal example data:

    .. code-block:: python

        import pandas as pd
        import champpy

        # Create example logbook data with synthetic journeys
        logbook_df = pd.DataFrame({
            'id_vehicle': [1, 1, 2],
            'dep_dt': pd.to_datetime(['2024-01-01 08:00', '2024-01-01 18:00', '2024-01-01 09:30']),
            'arr_dt': pd.to_datetime(['2024-01-01 12:00', '2024-01-01 22:00', '2024-01-01 17:30']),
            'dep_loc': [1, 2, 1],
            'arr_loc': [2, 1, 1],
            'distance': [25.5, 30.2, 18.0]
        })

        # Create example vehicle data
        vehicle_df = pd.DataFrame({
            'id_vehicle': [1, 2],
            'first_day': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'last_day': pd.to_datetime(['2024-01-02', '2024-01-02']),
            'id_cluster': [1, 1],
            'first_loc': [1, 1]
        })

        # Create mobility profiles
        mob_profiles = champpy.MobProfiles(input_logbooks_df=logbook_df,
                                   input_vehicles_df=vehicle_df)
    """

    def __init__(
        self,
        input_logbooks_df: pd.DataFrame,
        input_vehicles_df: pd.DataFrame | None = None,
        frozen: bool = False,
    ):
        """
        Initialize a MobProfiles object.

        The parameters are described in the class docstring.
        """
        # Initialize logbooks and vehicles
        self.logbooks = Logbooks(input_df=input_logbooks_df, frozen=frozen)
        if input_vehicles_df is not None:
            self.vehicles = Vehicles(input_df=input_vehicles_df)
            if self.vehicles.df["first_loc"].isnull().all():
                self.vehicles.set_first_loc_from_logbooks(self.logbooks)
        else:
            self.vehicles = Vehicles(frozen=False)
            self.vehicles.generate_vehicles_from_logbooks(self.logbooks)

        # Initialize clusters
        self.clusters = Clusters(self.vehicles, frozen=frozen)

        # Initialize locations
        self.locations = Locations(logbooks=self.logbooks, vehicles=self.vehicles, frozen=frozen)

        # set frozen after initialization
        self._frozen = frozen
        self._cleaned = False

        # Add observers to trigger functions in logbooks and clusters on vehicle changes
        self.vehicles._event_on_logbooks.add_observer(self.vehicles.delete_vehicles)
        self.vehicles._event_on_clusters.add_observer(self.clusters.update_clusters_from_vehicles)
        self.logbooks._event_on_locations.add_observer(self.locations.update_locations_from_logbooks_vehicles)

    def __copy__(self):
        """Create Copy of Instance that can be called by copy.copy(obj)"""
        return MobProfiles(self.logbooks.df, self.vehicles.df)

    def copy(self):
        """Create Copy of Instance"""
        return self.__copy__()

    def add_mob_profiles(
        self,
        input_mob_profiles: "MobProfiles",
        old_cluster_label: str = "Old",
        new_cluster_label: str = "New",
    ) -> None:
        """
        Add mobility data from another MobProfiles instance.
        The vehicles of the existing MobProfiles instance gets id_cluster = 1.
        The vehicles of the added MobProfiles instance gets id_cluster = 2.
        You can set labels for existing data using old_cluster_label and for added data using new_cluster_label.

        Parameters
        ----------
        input_mob_profiles : MobProfiles
                Another MobProfiles instance to add data from.
        old_cluster_label: str
                Label for existing data
        new_cluster_label: str
                Label for added data

        Examples
        --------
        Assuming `mob_profiles` exists (see :class:`MobProfiles` examples):

        .. code-block:: python

            # Create second dataset
            other_logbook_df = pd.DataFrame({...})
            other_mob_profiles = champpy.MobProfiles(other_logbook_df)

            # Add to existing mob_profiles
            mob_profiles.add_mob_profiles(input_mob_profiles=other_mob_profiles,
                                  old_cluster_label="Existing",
                                  new_cluster_label="Added")
        """
        if not isinstance(input_mob_profiles, MobProfiles):
            message = "other must be an instance of MobProfiles."
            logger.error(message)
            raise TypeError(message)

        # extract dataframes
        new_logbooks_df = input_mob_profiles.logbooks.df
        new_vehicles_df = input_mob_profiles.vehicles.df

        # Make sure vehicle IDs and clusters are unique across both datasets
        max_id_vehicle = self.vehicles.df["id_vehicle"].max() if not self.vehicles.df.empty else 0
        new_vehicles_df["id_vehicle"] += max_id_vehicle
        new_logbooks_df["id_vehicle"] += max_id_vehicle

        # Old data gets id_cluster = 1
        old_vehicles_df = self.vehicles.df
        old_vehicles_df["id_cluster"] = 1
        self.vehicles.df = old_vehicles_df

        # new data gets cluster = 2
        new_vehicles_df["id_cluster"] = 2

        # Add to logbooks and vehicles
        self.vehicles.add_vehicles(new_vehicles_df)
        self.logbooks.add_journeys(new_logbooks_df)

        # Set cluster labels
        clusters_df = self.clusters.df
        clusters_df.loc[clusters_df["id_cluster"] == 1, "label"] = old_cluster_label
        clusters_df.loc[clusters_df["id_cluster"] == 2, "label"] = new_cluster_label
        self.clusters.update_clusters(clusters_df)

        # Reindex IDs id_vehicles and id_journey after addition
        self.reindexing()

    @validate_call
    def reindexing(self, type: Literal["all", "id_journey", "id_vehicle", "id_cluster"] = "all") -> None:
        """
        Reindex of IDs in the MobProfiles instance (id_journey, id_vehicle, id_cluster).

        Parameters
        ----------
        type : Literal["all", "id_journey", "id_vehicle", "id_cluster"], optional
                Specifies which IDs to reindex. Default is "all".
                - "all": Reindex all IDs (id_journey, id_vehicle, id_cluster)
                - "id_journey": Reindex only journey IDs
                - "id_vehicle": Reindex only vehicle IDs
                - "id_cluster": Reindex only cluster IDs
        """

        if type in ["all", "id_vehicle"]:
            # Reindex vehicles based on logbooks starting from 1
            unique_vehicles = self.vehicles.df["id_vehicle"].unique()
            reindex_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_vehicles), start=1)}
            self.logbooks._df["id_vehicle"] = self.logbooks._df["id_vehicle"].map(reindex_map)
            self.vehicles._df["id_vehicle"] = self.vehicles._df["id_vehicle"].map(reindex_map)

        if type in ["all", "id_cluster"]:
            # Reindex cluster starting from 1
            unique_clusters = self.vehicles.df["id_cluster"].unique()
            cluster_map = {
                old_cluster: new_cluster for new_cluster, old_cluster in enumerate(sorted(unique_clusters), start=1)
            }
            self.vehicles._df["id_cluster"] = self.vehicles._df["id_cluster"].map(cluster_map)
            self.clusters._df["id_cluster"] = self.clusters._df["id_cluster"].map(cluster_map)

        if type in ["all", "id_journey"]:
            # Reindex id_journey
            unique_journeys = self.logbooks.df["id_journey"].unique()
            journey_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_journeys), start=1)}
            self.logbooks._df["id_journey"] = self.logbooks._df["id_journey"].map(journey_map)


class MobProfilesExtended:
    """
    Extended MobProfiles with additional attributes for modeling.

    Parameters
    ----------
    mob_profiles : MobProfiles
            Base MobProfiles instance.
    """

    def __init__(self, mob_profiles: MobProfiles, splitdays: bool = True, clustering: bool = True):

        if not isinstance(mob_profiles, MobProfiles):
            message = "mob_profiles must be an instance of MobProfiles class."
            logger.error(message)
            raise TypeError(message)

        # Predefine empty DataFrame with required columns
        self._df = pd.DataFrame(
            {
                "id_vehicle": pd.Series(dtype="int64"),
                "start_dt": pd.Series(dtype="datetime64[ns]"),
                "end_dt": pd.Series(dtype="datetime64[ns]"),
                "location": pd.Series(dtype="int64"),
                "speed": pd.Series(dtype="float64"),
            }
        )

        # Extend mob_profiles to include standing and non-driving vehicles
        self._extended_mob_profiles(mob_profiles)

        # Split multi-day rows if required
        self._split_multi_day_rows(splitdays=splitdays)

        # Join the 'id_cluster' column from t_vehicle into t_location
        if clustering:
            self._df = self._df.merge(
                mob_profiles.vehicles._df[["id_vehicle", "id_cluster"]],
                on="id_vehicle",
                how="left",
            )
            self._df["id_cluster"] = self._df["id_cluster"].astype("int64")
            self.labels_clusters = mob_profiles.clusters.df["label"].tolist()
            self.clusters = mob_profiles.clusters.df["id_cluster"].unique().tolist()
        else:
            self._df["id_cluster"] = 1
            self.labels_clusters = ["Total"]
            self.clusters = [1]

        self.labels_locations = mob_profiles.locations.df["label"].tolist()
        self.locations = mob_profiles.locations.df["location"].unique().tolist()

    @property
    def df(self) -> pd.DataFrame:
        """Get a copy of the extended MobProfiles DataFrame."""
        # Calculate distance and duration
        duration = (self._df["end_dt"] - self._df["start_dt"]).dt.total_seconds() / 3600  # in hours
        distance = self._df["speed"] * duration  # in km/h
        return self._df.copy().assign(duration=duration, distance=distance)

    def _extended_mob_profiles(self, mob_profiles: MobProfiles):
        """
        Create extended DataFrame with additional attributes.

        Returns
        -------
        pd.DataFrame
                Extended DataFrame.
        """
        # Logging
        logger.info("Extending MobProfiles")
        # convert automatically to uniform temporal resolution
        if mob_profiles.logbooks.temp_res is None:
            # find the minimum temporal resolution in hours
            min_res = mob_profiles.logbooks.df.apply(
                lambda row: (row["arr_dt"] - row["dep_dt"]).total_seconds() / 3600,
                axis=1,
            ).min()
            mob_profiles.logbooks.temp_res = min_res

        lb_df = mob_profiles.logbooks._df
        vehicles_df = mob_profiles.vehicles._df

        # determine first_loc for vehicles if nan
        if any(vehicles_df["first_loc"].isna()):
            vehicles_df.set_first_loc_from_logbooks(mob_profiles.logbooks)

        # Identify non-drivers
        mask_nondriver_vehicle = ~vehicles_df["id_vehicle"].isin(lb_df["id_vehicle"])
        n_nondriver_vehicle = mask_nondriver_vehicle.sum()

        # Create t_nondriver only if there are non-driver vehicles
        if n_nondriver_vehicle > 0:
            # Use first_loc if available, otherwise use default location 1
            nondriver_locations = vehicles_df.loc[mask_nondriver_vehicle, "first_loc"].astype("int64")

            nondriver_df = pd.DataFrame(
                {
                    "id_vehicle": vehicles_df.loc[mask_nondriver_vehicle, "id_vehicle"],
                    "start_dt": vehicles_df.loc[mask_nondriver_vehicle, "first_day"],
                    "end_dt": vehicles_df.loc[mask_nondriver_vehicle, "last_day"],
                    "location": nondriver_locations,
                    "speed": 0,
                }
            )
        else:
            nondriver_df = pd.DataFrame()

        # return if all vehicles are non-drivers
        if n_nondriver_vehicle == len(vehicles_df):
            self._df = nondriver_df.sort_values(by=["id_vehicle", "start_dt"]).reset_index(drop=True)
            return

        # Filter vehicles with journeys
        vehicle_df_drivers = vehicles_df.loc[~mask_nondriver_vehicle]

        # Find first and last track of each vehicle
        group = lb_df.groupby("id_vehicle")
        first_id_track = group["id_journey"].min()
        last_id_track = group["id_journey"].max()

        # Define rows for locations before the first trip
        start_df = pd.DataFrame(
            {
                "id_vehicle": vehicle_df_drivers["id_vehicle"],
                "start_dt": vehicle_df_drivers["first_day"],
                "end_dt": lb_df.dep_dt[lb_df["id_journey"].isin(first_id_track)].values,
                "location": lb_df.dep_loc[lb_df["id_journey"].isin(first_id_track)].values,
                "speed": 0,
            }
        )

        # Define rows for locations after the last trip
        end_df = pd.DataFrame(
            {
                "id_vehicle": vehicle_df_drivers["id_vehicle"],
                "start_dt": lb_df.arr_dt[lb_df["id_journey"].isin(last_id_track)].values,
                "end_dt": vehicle_df_drivers["last_day"] + pd.Timedelta(days=1),
                "location": lb_df.arr_loc[lb_df["id_journey"].isin(last_id_track)].values,
                "speed": 0,
            }
        )

        # Define rows for locations between trips
        standing_df = pd.DataFrame(
            {
                "id_vehicle": lb_df.id_vehicle[~lb_df["id_journey"].isin(last_id_track)].values,
                "start_dt": lb_df.arr_dt[~lb_df["id_journey"].isin(last_id_track)].values,
                "end_dt": lb_df.dep_dt[~lb_df["id_journey"].isin(first_id_track)].values,
                "location": lb_df.arr_loc[~lb_df["id_journey"].isin(last_id_track)].values,
                "speed": 0,
            }
        )

        # Define rows for location driving
        driving_df = pd.DataFrame(
            {
                "id_vehicle": lb_df["id_vehicle"],
                "start_dt": lb_df["dep_dt"],
                "end_dt": lb_df["arr_dt"],
                "location": 0,
                "speed": lb_df["distance"] / ((lb_df["arr_dt"] - lb_df["dep_dt"]).dt.total_seconds() / 3600),
            }
        )

        # Merge dataframes
        self._df = pd.concat([nondriver_df, start_df, standing_df, driving_df, end_df]).sort_values(
            by=["id_vehicle", "start_dt"]
        )
        self._df.reset_index(drop=True, inplace=True)

    def _split_multi_day_rows(self, splitdays: bool) -> pd.DataFrame:
        """
        Split multi-day rows in t_location into single-day rows.
        """
        if not splitdays:
            return

        # Split multi-day rows: vehicle is at one location over several days
        day_start = self._df["start_dt"].dt.floor("D")
        day_end = self._df["end_dt"].dt.floor("D")
        n_days = (day_end - day_start).dt.days + 1
        row_end_at_midnight = (self._df["end_dt"].dt.time == pd.Timestamp("00:00:00").time()) & (n_days > 1)

        # determine days per vehicle
        group = self._df.groupby("id_vehicle")
        first_day = group["start_dt"].min()
        last_day = group["end_dt"].max()
        days_per_vehicle = (last_day - first_day).dt.days

        # Abort if no multi-day rows exist
        if all(n_days == 1) or all(days_per_vehicle == 1):
            return self._df

        # New row for the last day of a multi-day row
        log_add_row_end = ~row_end_at_midnight & (n_days > 1)
        split_end_df = pd.DataFrame(
            {
                "id_vehicle": self._df.loc[log_add_row_end, "id_vehicle"],
                "start_dt": self._df.loc[log_add_row_end, "end_dt"].dt.floor("D"),
                "end_dt": self._df.loc[log_add_row_end, "end_dt"],
                "location": self._df.loc[log_add_row_end, "location"],
                "speed": self._df.loc[log_add_row_end, "speed"],
            }
        )

        # New rows for constant days in the middle: vehicle is at the same location over the whole day
        n_parking_days = n_days - 2
        n_parking_days[n_parking_days < 0] = 0
        parking_start_day = day_end[n_parking_days > 0] - pd.to_timedelta(n_parking_days[n_parking_days > 0], unit="D")
        parking_end_day = day_end[n_parking_days > 0]
        parking_days = [
            pd.date_range(start, end, inclusive="left") for start, end in zip(parking_start_day, parking_end_day)
        ]
        split_mid_df = pd.DataFrame(
            {
                "id_vehicle": np.repeat(
                    self._df.loc[n_parking_days > 0, "id_vehicle"].values,
                    n_parking_days[n_parking_days > 0],
                ),
                "start_dt": np.concatenate(parking_days),
                "end_dt": np.concatenate(parking_days) + pd.Timedelta(days=1),
                "location": np.repeat(
                    self._df.loc[n_parking_days > 0, "location"].values,
                    n_parking_days[n_parking_days > 0],
                ),
                "speed": 0,
            }
        )

        # Modify t_location for the first day of multi-day row
        self._df.loc[n_days > 1, "end_dt"] = day_start[n_days > 1] + pd.Timedelta(days=1)

        # Merge
        self._df = pd.concat([self._df, split_mid_df, split_end_df]).sort_values(by=["id_vehicle", "start_dt"])

        # reset index
        self._df.reset_index(drop=True, inplace=True)


class MobArray:
    """
    Mobility data in array format for efficient modeling.
    Child of MobProfilesExtended.
    """

    def __init__(self, mob_profiles: MobProfiles):
        # Logging
        logger.info("Creating MobArray from MobProfiles")
        # Check that all vehicles have same first_day and last_day
        n_first_days = mob_profiles.vehicles.df["first_day"].nunique()
        n_last_days = mob_profiles.vehicles.df["last_day"].nunique()
        if n_first_days != 1 or n_last_days != 1:
            message = "All vehicles in mob_profiles must have the same first_day and last_day to create MobArray."
            logger.error(message)
            raise ValueError(message)
        first_day = mob_profiles.vehicles.df["first_day"].iloc[0]
        last_day = mob_profiles.vehicles.df["last_day"].iloc[0]
        mob_profiles_ext_df = MobProfilesExtended(mob_profiles=mob_profiles, splitdays=True).df
        temp_res = mob_profiles.logbooks.temp_res
        dt_array, _ = get_datetime_array(start_date=first_day, end_date=last_day, temp_res=temp_res)
        # Get index in dt_array for start_dt and end_dt
        start_idx = pd.Series(
            np.searchsorted(dt_array, mob_profiles_ext_df["start_dt"].values),
            index=mob_profiles_ext_df.index,
        )
        end_idx = pd.Series(
            np.searchsorted(dt_array, mob_profiles_ext_df["end_dt"].values),
            index=mob_profiles_ext_df.index,
        )

        # Predefine arrays
        number_vehicles = mob_profiles.vehicles.number
        number_steps = len(dt_array)
        self.location = np.zeros((number_steps, number_vehicles), dtype=int)
        self.speed = np.zeros((number_steps, number_vehicles), dtype=float)
        self.distance = np.zeros((number_steps, number_vehicles), dtype=float)
        self.distance_distributed = np.zeros((number_steps, number_vehicles), dtype=float)
        self.speed_distributed = np.zeros((number_steps, number_vehicles), dtype=float)
        # Extract data into 1D arrays
        all_idx = np.concatenate([np.arange(s, e) for s, e in zip(start_idx, end_idx)])
        all_id_vehicles = np.concatenate(
            [np.full(e - s, vid) for vid, s, e in zip(mob_profiles_ext_df["id_vehicle"], start_idx, end_idx)]
        )
        all_locations = np.concatenate(
            [np.full(e - s, loc) for loc, s, e in zip(mob_profiles_ext_df["location"], start_idx, end_idx)]
        )
        all_speeds = np.concatenate(
            [np.full(e - s, spd) for spd, s, e in zip(mob_profiles_ext_df["speed"], start_idx, end_idx)]
        )
        all_distances = np.concatenate(
            [np.full(e - s, spd) for spd, s, e in zip(mob_profiles_ext_df["speed"], start_idx, end_idx)]
        )
        all_distances_distributed = np.concatenate(
            [
                np.full(e - s, dist / (e - s) if e > s else 0)
                for dist, s, e in zip(mob_profiles_ext_df["distance"], start_idx, end_idx)
            ]
        )
        all_speeds_distributed = np.concatenate(
            [np.full(e - s, spd) for spd, s, e in zip(mob_profiles_ext_df["speed"], start_idx, end_idx)]
        )

        # Convert into 2D arrays
        self.location[all_idx, all_id_vehicles - 1] = all_locations
        self.speed[all_idx, all_id_vehicles - 1] = all_speeds
        self.distance[all_idx, all_id_vehicles - 1] = all_distances
        self.distance_distributed[all_idx, all_id_vehicles - 1] = all_distances_distributed
        self.speed_distributed[all_idx, all_id_vehicles - 1] = all_speeds_distributed
        self.id_vehicle = np.arange(1, number_vehicles + 1)

        # Define departure array
        self.departure = np.zeros((number_steps, number_vehicles), dtype=bool)
        self.departure = self.speed > 0

        # Save datetime array
        self.datetime = dt_array
