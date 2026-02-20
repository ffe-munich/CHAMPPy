import logging
import pandas as pd
import numpy as np
import os
import pandera.pandas as pa

from importlib.resources import files
from dataclasses import dataclass, field
from rich.progress import track
from pandera.typing import Series
from itertools import product
from scipy.stats import beta

from champpy.utils.time_utils import get_day_index, TypeDays
from champpy.core.mobility.mobility_data import MobData, MobDataExtended
from champpy.core.mobility.mobility_validation import MobilityCharacteristics


# Define paths to data files using importlib.resources
DATA_DIR = files("champpy").joinpath("data")
PARAMS_DIR = DATA_DIR / "params.parquet"
PARAMS_INFO_DIR = DATA_DIR / "params_info.parquet"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserParamsParameterizer:
    """Class for user parameter used as input for the Parameterizer class."""

    description: str  # Description of the parameter set.
    vehicle_type: str  # Type of vehicle the parameters apply to.
    temp_res: float = 0.25  # Temporal resolution in hours
    typeday: TypeDays = field(
        default_factory=lambda: TypeDays(groups=[[0], [1], [2], [3], [4], [5], [6]])
    )  # List of weekday groups (0=Monday, 6=Sunday)
    speed_dist_edges_duration: list = field(
        default_factory=lambda: [0, 0.5, 1, 10]
    )  # List of speed distribution edges in hours

    def __post_init__(self):
        # Ensure temp_res is positive
        if self.temp_res <= 0:
            mssg = "temp_res must be positive. Got: %s", self.temp_res
            logger.error(mssg)
            raise ValueError(mssg)

        # Ensure speed_dist_edges_duration is sorted and positive
        edges = self.speed_dist_edges_duration
        if any(d < 0 for d in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
            mssg = f"speed_dist_edges_duration must be a sorted list of positive values. Got: {edges}"
            logger.error(mssg)
            raise ValueError(mssg)

        # Warning if speed_dist_edges_duration does not start with 0
        if edges[0] != 0:
            mssg = (
                f"speed_dist_edges_duration should start with 0 to include also trips with short duration. Got: {edges}"
            )
            logger.warning(mssg)

        # Ensure typeday is instance of TypeDays
        if not isinstance(self.typeday, TypeDays):
            mssg = f"typeday must be an instance of TypeDays class. Got: {type(self.typeday)}"
            logger.error(mssg)
            raise ValueError(mssg)


class ParamsSchema(pa.DataFrameModel):
    """Schema for calculated parameters for the mobility model."""

    id_params: int = pa.Field(ge=0, coerce=True)  # Unique identifier for the parameter set.
    id_cluster: int = pa.Field(ge=1, coerce=True, default=0)
    percentage: float = pa.Field(ge=0.0, le=100.0, coerce=True)
    speed_max: float = pa.Field(ge=0.0, coerce=True)
    weekdays: Series[object]  # List of weekday integers (0-6)
    transition_matrix: Series[object]  # 3D numpy array: (timesteps, locations, locations)
    speed_dist_param1: Series[object]  # List of speed distribution parameters (e.g. alpha)
    speed_dist_param2: Series[object]  # List of speed distribution parameters (e.g. beta)
    speed_dist_edges_duration: Series[object]  # List of speed distribution edges in hours

    class Config:
        strict = "filter"  # remove extra columns
        coerce = True  # enforce dtypes
        ordered = False  # don't enforce column order

    @pa.dataframe_check
    def check_transition_matrix(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure transition_matrix cells contain 3D numpy arrays with values between 0 and 1."""
        return df["transition_matrix"].apply(
            lambda x: isinstance(x, np.ndarray) and x.ndim == 3 and np.all((x >= 0) & (x <= 1))
        )

    @pa.dataframe_check
    def check_weekdays(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure weekdays cells contain lists of integers 0-6."""
        return df["weekdays"].apply(
            lambda x: isinstance(x, list) and all(isinstance(d, int) and 0 <= d <= 6 for d in x)
        )

    @pa.dataframe_check
    def check_speed_dist_params(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure speed distribution parameter cells contain lists of floats."""
        bool_param2 = df["speed_dist_param2"].apply(
            lambda x: isinstance(x, list) and all(isinstance(d, (float, np.floating)) for d in x)
        )
        bool_param1 = df["speed_dist_param1"].apply(
            lambda x: isinstance(x, list) and all(isinstance(d, (float, np.floating)) for d in x)
        )
        bool_edges = df["speed_dist_edges_duration"].apply(
            lambda x: isinstance(x, list) and all(isinstance(d, (float, np.floating)) for d in x)
        )
        return bool_param1 & bool_param2 & bool_edges


class ParamsInfoSchema(pa.DataFrameModel):
    # TODO REMOVE no longer needed
    """Schema for calculated parameter information for the mobility model."""
    id_params: int = pa.Field(ge=0, coerce=True)  # Unique identifier for the parameter set.
    description: str = pa.Field(coerce=True)  # Description of the parameter set.
    vehicle_type: str = pa.Field(coerce=True)  # Type of vehicle the parameters apply to.
    temp_res: float = pa.Field(coerce=True)  # Temporal resolution of the mobility data in hours.
    annual_km: float = pa.Field(coerce=True)  # Annual kilometers driven as reference.
    locations: Series[object]  # Numpy array of location IDs
    number_typedays: int = pa.Field(ge=1, le=7, coerce=True)  # Number of typedays used in the parameterization.
    number_clusters: int = pa.Field(ge=1, coerce=True)  # Number of clusters used in the parameterization.
    labels_locations: Series[object]  # List of location labels corresponding to location IDs
    labels_clusters: Series[object]  # List of cluster labels corresponding to cluster IDs
    created_user: str = pa.Field(coerce=True)  # User who created the parameter set.
    created_dt: pd.Timestamp = pa.Field(coerce=True)  # Datetime when the parameter set was created.

    class Config:
        """Configuration for ParamsInfoSchema."""

        strict = "filter"  # remove extra columns
        coerce = True  # enforce dtypes
        ordered = False  # don't enforce column order

    @pa.dataframe_check
    def check_locations(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure locations cells contain numpy arrays of integers >= 1."""
        return df["locations"].apply(
            lambda x: isinstance(x, np.ndarray) and all(isinstance(d, int) and d >= 1 for d in x)
        )


@dataclass
class ParamsInfo:
    id_params: int  # Unique identifier for the parameter set.
    description: str  # Description of the parameter set.
    vehicle_type: str  # Type of vehicle the parameters apply to.
    temp_res: float  # Temporal resolution of the mobility data in hours.
    annual_km: float  # Annual kilometers driven as reference.
    locations: list[int]  # List of locations occurring in the mobility data.
    share_of_time_at_locations: list[float]  # Share of time the vehicles spend at each location.
    number_typedays: int  # Number of typedays used in the parameterization.
    number_clusters: int  # Number of clusters used in the parameterization.
    labels_locations: list[str]  # List of location labels corresponding to location IDs
    labels_clusters: list[str]  # List of cluster labels corresponding to cluster IDs
    created_user: str = field(
        default_factory=lambda: os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    )  # User who created the parameter set.
    created_dt: pd.Timestamp = field(default_factory=pd.Timestamp.now)  # Datetime when the parameter set was created.


@dataclass(frozen=True)
class ModelParams:
    """Class for calculated parameters used in the mobility model."""

    df: pd.DataFrame  # DataFrame with calculated parameters
    info: ParamsInfo  # Information about the parameter set


class Parameterizer:
    """Class for parameterization of the mobility model."""

    def __init__(self, user_params: UserParamsParameterizer):
        """Initialize Parameterization with parameter and info DataFrames.

            Args:
                    user_params (UserParamsParameterizer): User-defined parameters for the model.
        ref_data (MobData): Cleaned Mobility data as reference.
        """
        # Store user parameters
        self.user_params = user_params
        # Internal placeholder for calculation of parameters DataFrame
        self._params_df: pd.DataFrame = pd.DataFrame()
        # Initalize placehlders for temporary variables
        self._unique_locations: list[int] = []

    def calc_params(self, ref_data: MobData) -> ModelParams:
        """Calculate parameters for the mobility model."""

        # Abort if mob_data is not cleaned
        if not ref_data._is_cleaned:
            mssg = "ref_data must be cleaned mobility data. Please clean mobility data using MobDataCleaner before parameterization."
            logger.error(mssg)
            raise ValueError(mssg)

        logger.info("Starting parameterization of the mobility model.")

        # Create info DataFrame
        params_info = self._create_info(ref_data)

        # initialize params DataFrame
        number_cluster = ref_data.vehicles.df["id_cluster"].nunique()
        number_typeday = len(self.user_params.typeday.groups)
        number_rows = number_cluster * number_typeday
        clusters = ref_data.vehicles.df["id_cluster"].unique()

        # Create weekdays by repeating typeday for each cluster (keep as lists)
        weekdays_repeated = self.user_params.typeday.groups * number_cluster

        # edges als float-Liste
        edges_float = [float(e) for e in self.user_params.speed_dist_edges_duration]
        self._params_df = pd.DataFrame(
            {
                "id_params": [params_info.id_params] * number_rows,
                "id_cluster": np.repeat(clusters, number_typeday),
                "weekdays": weekdays_repeated,
                "percentage": np.zeros(number_rows),
                "speed_max": np.zeros(number_rows),
                "transition_matrix": [None] * number_rows,  # Will be filled with 3D arrays
                "speed_dist_param1": [None] * number_rows,
                "speed_dist_param2": [None] * number_rows,
                "speed_dist_edges_duration": [edges_float] * number_rows,
            }
        )

        # Calculate parameters
        self._calc_all_parameters(ref_data)

        # Validate params DataFrame
        validated_params_df = ParamsSchema.validate(self._params_df)

        # Return result as ModelParams dataclass
        return ModelParams(df=validated_params_df, info=params_info)

    def _create_info(self, ref_data: MobData) -> ParamsInfo:
        """Get parameter information DataFrame."""
        mob_char = MobilityCharacteristics(ref_data, typedays=TypeDays(groups=[[0, 1, 2, 3, 4, 5, 6]]))
        # Create info DataFrame
        params_info = ParamsInfo(
            id_params=0,  # TODO Placeholder, should be unique identifier
            description=self.user_params.description,
            vehicle_type=self.user_params.vehicle_type,
            temp_res=self.user_params.temp_res,
            annual_km=(mob_char.df.loc[0, "daily_kilometrage"] * 365).round(3),
            locations=mob_char.df.loc[0, "locations"],
            share_of_time_at_locations=mob_char.df.loc[0, "share_of_time_at_locations"].round(3),
            number_typedays=len(self.user_params.typeday.groups),
            number_clusters=ref_data.vehicles.df["id_cluster"].nunique(),
            labels_locations=ref_data.locations.df["label"].tolist(),
            labels_clusters=ref_data.clusters.df["label"].tolist(),
        )
        return params_info

    def _calc_all_parameters(self, ref_data: MobData):
        """Calculate parameters for the mobility model."""

        # Determine percentage of clusters based on number of days per cluster in ref_data
        self._calc_percentage_clusters(ref_data)

        # Extend reference data
        ref_data_df_ext = MobDataExtended(ref_data).df

        # add weekday and index columns
        temp_res = ref_data.logbooks.temp_res
        ref_data_df_ext["weekday"] = ref_data_df_ext["start_dt"].dt.dayofweek  # Monday=0, Sunday=6
        ref_data_df_ext["start_index"] = get_day_index(ref_data_df_ext["start_dt"], temp_res)
        ref_data_df_ext["end_index"] = get_day_index(ref_data_df_ext["end_dt"], temp_res)

        # Reindex locations
        ref_data_df_ext = self._reindex_locations(ref_data_df_ext)

        # Loop over each row in params DataFrame
        number_rows = self._params_df.shape[0]
        for idx in track(range(number_rows), description="Parameterization:"):
            cluster = self._params_df.at[idx, "id_cluster"]
            weekdays = self._params_df.at[idx, "weekdays"]
            logger.debug(f"Calculating parameters for cluster {cluster}, weekdays {weekdays}")

            # Filter ref_data for current cluster and weekdays
            mask_cluster = ref_data_df_ext["id_cluster"] == cluster
            mask_weekdays = ref_data_df_ext["weekday"].isin(weekdays)
            ref_data_df_ext_filtered = ref_data_df_ext[mask_cluster & mask_weekdays]

            # Calculate parameters for this cluster and weekdays
            self._calc_parameters_for_idx(ref_data_ext=ref_data_df_ext_filtered, idx=idx)

    def _calc_percentage_clusters(self, ref_data: MobData):
        """Calculate percentage of days per cluster."""
        vehicles_df = ref_data.vehicles.df
        vehicles_df["number_days"] = vehicles_df["last_day"] - vehicles_df["first_day"] + pd.Timedelta(days=1)
        number_days_total = vehicles_df["number_days"].sum()
        number_days_cluster = vehicles_df.groupby("id_cluster")["number_days"].sum()
        percentage_cluster = number_days_cluster / number_days_total * 100
        self._params_df["percentage"] = self._params_df["id_cluster"].map(percentage_cluster).values

    def _calc_parameters_for_idx(self, ref_data_ext: pd.DataFrame, idx: int):
        """Calculate parameters for the parameterization."""
        self._calc_transition_matrix(ref_data_ext, idx)
        self._calc_speed_distribution(ref_data_ext, idx)

    def _reindex_locations(self, ref_data_ext: pd.DataFrame) -> pd.DataFrame:
        """Reindex locations for the parameterization."""
        # save unique locations excluding zero
        unique_locations = ref_data_ext["location"].unique()
        unique_locations_nozero = unique_locations[unique_locations != 0]
        locations_sorted = sorted(unique_locations.tolist())
        self._unique_locations = locations_sorted

        # Reindex locations to consecutive integers starting from 1, keep 0 as is
        location_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_locations_nozero, start=1)}
        ref_data_ext.loc[:, "location"] = ref_data_ext["location"].map(location_mapping).fillna(0).astype(int)
        return ref_data_ext

    def _calc_transition_matrix(self, ref_data_ext: pd.DataFrame, idx: int):
        """Calculate transition matrices for the mobility model."""
        # Throw error if ref_data_ext is empty
        if ref_data_ext.empty:
            mssg = f"There is no data for cluster {self._params_df.at[idx, 'id_cluster']} and weekdays {self._params_df.at[idx, 'weekdays']}. Cannot calculate transition matrix."
            logger.error(mssg)
            raise ValueError(mssg)

        # Predefine required variables
        unique_location = np.arange(len(self._unique_locations))
        unique_index_day = np.arange(int(24 / self.user_params.temp_res))
        n_steps_per_day = len(unique_index_day)
        n_locations = len(unique_location)

        # predefine transition DataFrame: alle Kombinationen von day_index, start_loc, end_loc
        combinations = list(product(unique_index_day, unique_location, unique_location))
        transition_df = pd.DataFrame(combinations, columns=["day_index", "start_loc", "end_loc"])
        transition_df["count"] = 0
        transition_df.set_index(["day_index", "start_loc", "end_loc"], inplace=True)

        # Determine counts of timesteps without transitions between locations: vehicle stays at same location
        starts = ref_data_ext["start_index"].values
        ends = ref_data_ext["end_index"].values - 1
        ends[ends < 0] = n_steps_per_day - 1  # handle end index 0 as last index
        locs = ref_data_ext["location"].values
        lengths = ends - starts + 1
        mask = lengths > 0
        if np.any(mask):
            all_day_indices = np.concatenate([np.arange(s, e + 1) for s, e in zip(starts[mask], ends[mask])])
            all_locs = np.repeat(locs[mask], lengths[mask])
            records = np.column_stack((all_day_indices, all_locs, all_locs))
            non_trans_df = pd.DataFrame(records, columns=["day_index", "start_loc", "end_loc"])
            non_trans_df = non_trans_df.astype({"day_index": int, "start_loc": int, "end_loc": int})
            non_trans_df_grouped = (
                non_trans_df.groupby(["day_index", "start_loc", "end_loc"]).size().reset_index(name="count")
            )
            non_trans_df_grouped.set_index(["day_index", "start_loc", "end_loc"], inplace=True)
            del non_trans_df  # save memory

        # Determine counts of timesteps with transitions between locations: vehicledrives from one location to another
        mask_not_first_row_of_vehicle = ref_data_ext["start_dt"] != ref_data_ext.groupby("id_vehicle")[
            "start_dt"
        ].transform("min")
        start_indexs = ref_data_ext.loc[mask_not_first_row_of_vehicle, "start_index"].values
        end_loc = ref_data_ext.loc[mask_not_first_row_of_vehicle, "location"].values
        start_loc = ref_data_ext["location"].shift(+1).values
        start_loc = start_loc[mask_not_first_row_of_vehicle]  # remove first rows of vehicles
        trans_df = pd.DataFrame({"day_index": start_indexs, "start_loc": start_loc, "end_loc": end_loc})
        trans_df = trans_df.astype({"day_index": int, "start_loc": int, "end_loc": int})
        trans_df_grouped = trans_df.groupby(["day_index", "start_loc", "end_loc"]).size().reset_index(name="count")
        trans_df_grouped.set_index(["day_index", "start_loc", "end_loc"], inplace=True)
        del trans_df  # save memory

        # Merge counts of non-transition and transition DataFrames
        transition_df["count"] = transition_df["count"].add(non_trans_df_grouped["count"], fill_value=0)
        transition_df["count"] = transition_df["count"].add(trans_df_grouped["count"], fill_value=0)
        transition_df = transition_df.reset_index()

        # Berechne total_counts pro day_index
        total_counts = transition_df.groupby(["day_index", "start_loc"])["count"].sum()
        transition_df = transition_df.merge(
            total_counts.rename("total_count"), on=["day_index", "start_loc"], how="left"
        )

        # Calculate transition probabilities
        transition_df["probability"] = transition_df["count"] / transition_df["total_count"]
        transition_df.fillna(0, inplace=True)

        # Reshape to 3D numpy array
        tm = np.zeros((n_steps_per_day, n_locations, n_locations))
        for _, row in transition_df.iterrows():
            day_index = int(row["day_index"])
            start_loc = int(row["start_loc"])
            end_loc = int(row["end_loc"])
            tm[day_index, start_loc, end_loc] = row["probability"]

        self._params_df.at[idx, "transition_matrix"] = tm

    def _calc_speed_distribution(self, ref_data_ext: pd.DataFrame, idx: int):
        """Calculate speed distribution parameters using a Beta distribution."""

        # Get variables
        lb_speed_df = ref_data_ext[ref_data_ext["location"] == 0][["speed", "duration"]]
        edges_duration = self._params_df.at[idx, "speed_dist_edges_duration"]

        # Extract durations and speeds for different duration bins
        speeds_binned = []
        for i in range(len(edges_duration) - 1):
            lower_edge = edges_duration[i]
            upper_edge = edges_duration[i + 1]
            mask = (lb_speed_df["duration"] >= lower_edge) & (lb_speed_df["duration"] < upper_edge)
            speeds_binned.append(lb_speed_df.loc[mask, "speed"].values)

        # Normalize speeds to [0, 1] for Beta distribution fitting
        max_speed = lb_speed_df["speed"].max() * 1.1  # add 10% margin
        self._params_df.at[idx, "speed_max"] = max_speed
        speeds_binned_normalized = [speeds / max_speed for speeds in speeds_binned]

        # Fit Beta distribution to each bin
        param1_list = []
        param2_list = []
        for speeds_binned_i in speeds_binned_normalized:
            # Plot die aktuelle Bin-Verteilung
            if len(speeds_binned_i) < 2:
                # Not enough data to fit distribution
                param1_list.append(np.nan)
                param2_list.append(np.nan)
            else:
                params = beta.fit(speeds_binned_i, floc=0, fscale=1)
                param1_list.append(params[0])
                param2_list.append(params[1])

        self._params_df.at[idx, "speed_dist_param1"] = [float(x) for x in param1_list]
        self._params_df.at[idx, "speed_dist_param2"] = [float(x) for x in param2_list]


class ParamsLoader:
    """Class for loading pre-calculated parameters for the mobility model."""

    def __init__(self, user_name: str = None):
        """Initialize ParameterLoader with database connection.

        Args:
            db: Database connection object.
        """
        if user_name is None:
            user_name: str = field(
                default_factory=lambda: os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
            )
        else:
            self.user_name = user_name

    def load_info(self) -> pd.DataFrame:
        """Load info DataFrame from params_info.parquet, return empty DataFrame if not found."""
        if not PARAMS_INFO_DIR.exists():
            return pd.DataFrame()
        return pd.read_parquet(PARAMS_INFO_DIR)

    def load_params(self, id_params: int = None) -> ModelParams:
        """Load existing ModelParams."""

        logger.info("Load parameters with id_params=%s.", id_params)
        
        # Load info for existing params
        info_df = self.load_info()
        if info_df.empty:
            mssg = "There are no existing parameters."
            logger.error(mssg)
            raise ValueError(mssg)
        if id_params is not None and id_params not in info_df["id_params"].values:
            mssg = f"No parameters found for id_params = {id_params}. \nCheck ParamsLoader.load_info for available parameters."
            logger.error(mssg)
            raise ValueError(mssg)

        # Load params DataFrame
        params_df = self._load_only_params(id_params)

        # convert transition
        params_df = self._convert_params_df_list2tm(params_df)

        # Create ParamsInfo
        info_row = info_df[info_df["id_params"] == id_params].iloc[0]
        params_info = ParamsInfo(
            id_params=info_row["id_params"],
            description=info_row["description"],
            vehicle_type=info_row["vehicle_type"],
            temp_res=info_row["temp_res"],
            annual_km=info_row["annual_km"],
            locations=info_row["locations"],
            share_of_time_at_locations=info_row["share_of_time_at_locations"],
            number_typedays=info_row["number_typedays"],
            number_clusters=info_row["number_clusters"],
            labels_locations=info_row["labels_locations"],
            labels_clusters=info_row["labels_clusters"],
            created_user=info_row["created_user"],
            created_dt=info_row["created_dt"],
        )
        return ModelParams(df=params_df, info=params_info)

    def _load_only_params(self, id_params: int = None) -> pd.DataFrame:
        """Load only params DataFrame from params.parquet."""
        # Load params DataFrame
        if id_params is not None:
            params_df = pd.read_parquet(PARAMS_DIR, filters=[("id_params", "==", id_params)])
        else:
            params_df = pd.read_parquet(PARAMS_DIR)
        return params_df

    def _save_params(self, params: ModelParams) -> int:
        """Save calculated parameters."""

        # Load info for existing params
        info_df = self.load_info()

        # Save info DataFrame
        if info_df.empty:
            # add new id_params
            new_id = 1
            params.info.id_params = new_id
            # create info_df from ParamsInfo
            info_df = pd.DataFrame([vars(params.info)])
        else:
            # check if params with same description already exist
            mask_existing = info_df["description"] == params.info.description
            if mask_existing.any():
                mssg = (
                    f"There are already Parameters with description '{params.info.description}'. "
                    f"Please define a unique description for the new parameters or use id_params = {info_df[mask_existing]['id_params'].values[0]} instead."
                )
                logger.error(mssg)
                raise ValueError(mssg)
            # round annual_km to avoid floating point issues
            info_df["annual_km"] = info_df["annual_km"].round(3)
            # check with the same info excluding description and created_user, created_dt
            cols_to_check = [
                col for col in info_df.columns if col in ["temp_res", "annual_km", "number_typedays", "number_clusters"]
            ]
            mask_existing_info = (info_df[cols_to_check] == pd.Series(vars(params.info))[cols_to_check]).all(axis=1)
            if mask_existing_info.any():
                id_val = info_df[mask_existing_info]["id_params"].values[0]
                mssg = (
                    f"The are already Parameters with the same properties. "
                    f"Please use id_params = {id_val} instead of creating new entries. "
                    f"Check: \n{info_df[mask_existing_info]}"
                )
                logger.error(mssg)
                raise ValueError(mssg)

            # assign new id_params
            new_id = info_df["id_params"].max() + 1
            params.info.id_params = new_id
            # append new info
            info_df = pd.concat([info_df, pd.DataFrame([vars(params.info)])], ignore_index=True)

        # Save info DataFrame
        info_df.to_parquet(PARAMS_INFO_DIR, index=False)

        # Add id_params to params DataFrame
        params.df["id_params"] = new_id

        # Convert transition matrix np arrays to lists for saving
        params_df = params.df.copy()
        params_df = self._convert_params_df_tm2list(params_df)

        # load existing params DataFrame
        if PARAMS_DIR.exists():
            params_existing_df = self._load_only_params()
            # append new params
            params_df = pd.concat([params_existing_df, params_df], ignore_index=True)

        # Save params DataFrame
        params_df.to_parquet(PARAMS_DIR, index=False)

        return new_id

    def _delete_params(self, id_params: int):
        """Delete parameters with given id_params."""
        # Load info DataFrame
        info_df = self.load_info()
        if info_df.empty or id_params not in info_df["id_params"].values:
            mssg = f"No parameters found for id_params = {id_params}."
            logger.error(mssg)
            raise ValueError(mssg)

        # Delete from info DataFrame
        info_df = info_df[info_df["id_params"] != id_params]
        info_df.to_parquet(PARAMS_INFO_DIR, index=False)

        # Load params DataFrame
        params_df = self._load_only_params()

        # Delete from params DataFrame
        params_df = params_df[params_df["id_params"] != id_params]
        params_df.to_parquet(PARAMS_DIR, index=False)

    @classmethod
    def deep_to_numpy(cls, arr):
        """Recursively convert nested lists or object arrays to float numpy arrays."""
        # If arr is an object array, recursively convert to float arrays
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            # Recursively process each row/layer
            return np.array([cls.deep_to_numpy(x) for x in arr], dtype=float)
        # If arr is a list, recursively process
        if isinstance(arr, list):
            return np.array([cls.deep_to_numpy(x) for x in arr], dtype=float)
        # If arr is already float or int, just return
        return arr

    @classmethod
    def _convert_params_df_list2tm(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Convert transition_matrix column from list/nested array to 3D np.ndarray."""
        df["transition_matrix"] = df["transition_matrix"].apply(cls.deep_to_numpy)
        return df

    @staticmethod
    def _convert_params_df_tm2list(df: pd.DataFrame) -> pd.DataFrame:
        """Convert transition_matrix column from np.ndarray to list."""
        df["transition_matrix"] = df["transition_matrix"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        return df
