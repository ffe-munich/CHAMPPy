import logging
import pandas as pd
import numpy as np
import os
import pandera.pandas as pa

from typing import Tuple
from dataclasses import dataclass, field
from rich.progress import track
from pandera.typing import Series
from scipy.stats import beta, mode
from scipy import sparse

from champpy.core.mobility_data import MobData, LogbookSchema
from champpy.core.parameterization import ModelParams
from champpy.utils.time_utils import get_day_index, TypeDays


logger = logging.getLogger(__name__)

@dataclass
class UserParamsMobModel:
    """
    Dataclass for user parameters for the mobility model.
    Args:
        n_clusters: Number of clusters to use in the mobility model.
    """
    number_vehicles: int = 50 # Number of vehicles to generate mobility profiles for
    start_date: pd.Timestamp = pd.Timestamp("2025-01-01") # Start date for the mobility profiles
    end_date: pd.Timestamp = pd.Timestamp("2025-12-31") # End date for the mobility profiles
    random_seed: int = 1 # Random seed for reproducibility

    def __post_init__(self):
        # Validate number of vehicles: positive integer
        if self.number_vehicles < 1:
            message = "Number of vehicles must be at least 1."
            logging.error(message)
            raise ValueError(message)
        
        # Validate start_date and end_date format: start must be before end and at least one day apart
        if self.start_date.date() >= self.end_date.date():
            message = "Start date must be at least one day before end date."
            logging.error(message)
            raise ValueError(message)
        

class MobModel:
    """
    Class for the model that creates mobility profiles (MobProfiles).
    Args:
        params: ModelParams dataclass containing mobility model parameters.
    """
    def __init__(self, model_params: ModelParams):
        self.model_params = model_params


    def generate_mob_profiles(self, user_params: UserParamsMobModel) -> MobData:
        
        # Set random seed
        np.random.seed(user_params.random_seed)

        # Predefine variables
        self._predefine_vars(user_params=user_params)
        previous_start = np.zeros((self._number_vehicles,), dtype=int)

        # Define a possible location for the first step based on the transition matrix
        possible_start_locations = [1]# TODO : determine based on transition matrix
        self._location_array[0, :] = np.random.choice(possible_start_locations, size=self._number_vehicles, replace=True)
    
        # Add rich progress bar for vehicle loop
        for t in track(range(1, self._number_steps), description="Generating mobility profiles..."):
            # Determine new location based on transition matrix
            self._generate_location(t)
            for v in range(self._number_vehicles):
                location_t = self._location_array[t, v]
                location_tminus1 = self._location_array[t-1, v]
                # Identify start and end of journeys
                if location_t == 0 and location_tminus1 != 0 and t != self._number_steps - 1:
                    # Start of a new journey
                    self._start_journey_array[t, v] = True
                    previous_start[v] = t

                elif (location_t >0 or t == (self._number_steps - 1)) and location_tminus1 == 0: 
                    # End of the current journey
                    journey_duration_h = (t - previous_start[v]) * self.model_params.info.temp_res
                    self._duration_array[previous_start[v],v] = journey_duration_h
                    if t == (self._number_steps - 1):
                        # Intervene if journey ends at the last time step
                        # Set location to most frequent location:
                        most_frequent_location = mode(self._location_array[:, v])[0]
                        self._location_array[t, v] = most_frequent_location

        # Claculate speed and distance arrays based on start_journey_array and duration_array
        self._generate_speed_and_distance()

        # Convert arrays to Mobdata instance
        mob_profiles = self._convert_arrays2mob_profiles()

        return mob_profiles
    
    def _predefine_vars(self, user_params: UserParamsMobModel) -> None:
        """
        Predefine variables for the mobility model.
        Args:
            user_params: UserParamsMobModel dataclass containing user parameters.
        """

        self._number_vehicles = user_params.number_vehicles

        # Determine datetime array
        temp_res = self.model_params.info.temp_res
        start_dt = user_params.start_date.normalize()
        end_dt = user_params.end_date.normalize() + pd.Timedelta(days=1) - pd.Timedelta(hours=temp_res)
        frequency = f"{self.model_params.info.temp_res}H"
        self._dt_array = pd.date_range(start=start_dt, end=end_dt, freq=frequency)
        self._number_steps = len(self._dt_array)

        self._index_day_array = get_day_index(self._dt_array, self.model_params.info.temp_res)
        weekday_array = self._dt_array.weekday
        mask_cluster0 = self.model_params.df["cluster"] == 0
        typedays = [[int(i) for i in list(x)] for x in self.model_params.df.loc[mask_cluster0, "weekdays"]]
        typedays_array = TypeDays(typedays).weekday2typeday(weekday_array)
        first_weekday = self.model_params.df["weekdays"].apply(lambda x: x[0])
        self.model_params.df["typeday"] = TypeDays(typedays).weekday2typeday(first_weekday)

        # Store transition matrices in one array for faster access
        self._tm_array = np.stack(self.model_params.df["transition_matrix"].to_numpy())

        # generate random number to determine new location
        self._rand1_array = np.random.rand(self._number_steps, self._number_vehicles)

        # split vehicles into clusters
        cluster_array = self._split_vehicles_per_cluster(self._number_vehicles)
    
        # Determine index of parameters based on cluster_array and typedays_array
        self._index_params_array = (
            self.model_params.df.reset_index()
            .pivot(index="typeday", columns="cluster", values="index")
            .loc[typedays_array, cluster_array]
            .to_numpy()
        )
        
        # Initialize arrays for location, speed, distance, duration (sparse for speed, distance, duration)
        self._location_array = np.zeros((self._number_steps, self._number_vehicles), dtype=int) # location of vehicles
        self._speed_array = np.zeros((self._number_steps, self._number_vehicles), dtype=float) # speed of journeys (sparse)
        self._distance_array = np.zeros((self._number_steps, self._number_vehicles), dtype=float) # distance of journeys (sparse)
        self._duration_array = np.zeros((self._number_steps, self._number_vehicles), dtype=float) # duration of journeys (sparse)
        self._start_journey_array = np.zeros((self._number_steps, self._number_vehicles), dtype=bool) # start of journeys

    def _split_vehicles_per_cluster(self, number_vehicles: int) -> np.ndarray:
        """
        Split the total number of vehicles into clusters based on the model parameters.
        Args:
            number_vehicles: Total number of vehicles to split.
        
        Returns:
            dict[int, int]: Dictionary with cluster ID as key and number of vehicles as value.
        """
        percentages_per_cluster = self.model_params.df.groupby("cluster")["percentage"].first()
        vehicles_per_cluster = (percentages_per_cluster / 100 * number_vehicles).round().astype(int)
        rest = number_vehicles - vehicles_per_cluster.sum()
        if rest > 0:
            # Assign remaining vehicles to the largest cluster
            largest_cluster = vehicles_per_cluster.idxmax()
            vehicles_per_cluster[largest_cluster] += rest
        
        # Create array with cluster IDs for each vehicle
        cluster_array = np.zeros(number_vehicles, dtype=int)
        current_idx = 0
        for cluster_id, n_vehicles in vehicles_per_cluster.items():
            cluster_array[current_idx:current_idx + n_vehicles] = cluster_id
            current_idx += n_vehicles

        return cluster_array
    
    def _generate_location(self, t) -> None:
        """
        Generate the locations for all vehicle for one timestep based on the transition matrix.
        Args:
            t: Time step
        """
        # Parameter index, previous location and day index for all vehicles
        params_idx = self._index_params_array[t, :]
        loc_tminus1 = self._location_array[t-1, :]
        day_idx = self._index_day_array[t]

        # Transition vectors and cumulative transition vectors for all vehicles
        trans_vecs = self._tm_array[params_idx, day_idx, loc_tminus1, :]
        cum_trans_vecs = np.cumsum(trans_vecs, axis=1)

        # Random numbers for all vehicles
        rand_t = self._rand1_array[t, :]

        # New locations for all vehicles
        location_t = np.sum(rand_t[:, None] > cum_trans_vecs, axis=1)
        self._location_array[t, :] = location_t

    def _generate_speed_and_distance(self) -> None:
        """
        Generate speed and distance arrays based on start_journey_array and duration_array.
        """
        # generate speed and distance for all fields where duration > 0
        mask_start = self._start_journey_array
        index_params_jarray = self._index_params_array[mask_start]
        edges_duration = self.model_params.df.loc[0, "speed_dist_edges_duration"]

        # only use the array entries where journeys start: journeys array (jarray)
        duration_jarray = self._duration_array[mask_start]
        number_journeys = duration_jarray.shape[0]

        # identify idx_duration for all journeys
        idx_duration_jarray = np.searchsorted(edges_duration, duration_jarray, side="right") - 1
        max_index_duration = len(edges_duration)-2
        idx_duration_jarray = np.minimum(idx_duration_jarray, max_index_duration) # cap at max index

        # Get speed distribution parameters for all journeys (vectorized, no loop)
        speed_param1_full = np.array(self.model_params.df["speed_dist_param1"].to_list())  # shape: (n_paramsets, n_bins)
        speed_param1_jarray = speed_param1_full[index_params_jarray, idx_duration_jarray]
        speed_param2_full = np.array(self.model_params.df["speed_dist_param2"].to_list())  # shape: (n_paramsets, n_bins)
        speed_param2_jarray = speed_param2_full[index_params_jarray, idx_duration_jarray]
        speed_max_array = self.model_params.df["speed_max"].to_numpy()[index_params_jarray]

        # Generate random numbers for all journeys where duration > 0
        rand2_array = np.random.rand(number_journeys )

        # Generate speed for all journeys (vectorized, no loop)
        speed_jarray = beta.ppf(rand2_array, speed_param1_jarray, speed_param2_jarray) * speed_max_array

        # Generate distance array
        distance_jarray = speed_jarray * duration_jarray

        # Set speed and distance values back to full arrays
        self._speed_array[mask_start] = speed_jarray
        self._distance_array[mask_start] = distance_jarray
    
    def _generate_speed(self, t, v, index_start, duration) -> None:
        """
        Generate a speed value from the specified distribution parameters.
        Args:
            t: Time step index.
            v: Vehicle index.
            index_start: Index of the start of the journey.
            duration: Duration of the journey in hours.
        """
        # Identify the index of the speed params based  on the duration
        index_params = self._index_params_array[t, v]
        edges_duration = self.model_params.df.loc[index_params, "speed_dist_edges_duration"]
        max_index_duration = len(edges_duration)-2
        index_duration = np.searchsorted(edges_duration, duration, side="right") - 1
        index_duration = min(index_duration, max_index_duration)

        # Get speed distribution parameters from params from params dataframe
        speed_param1 = self.model_params.df.loc[index_params, "speed_dist_param1"][index_duration]
        speed_param2 = self.model_params.df.loc[index_params, "speed_dist_param2"][index_duration]
        speed_max = self.model_params.df.loc[index_params, "speed_max"]

        # Generate random number to determine speed
        rand2 = np.random.rand()
        speed = beta.ppf(rand2, speed_param1, speed_param2) * speed_max

        # Set speed for the journey
        self._speed_array[index_start, v] = speed

    def _convert_arrays2mob_profiles(self) -> MobData:
        """
        Convert the generated arrays to a pandas DataFrame representing the logbook.
        Returns:
            pd.DataFrame: DataFrame containing the logbook data.
        """
        # get rows and cols of journeys
        rows, cols = np.nonzero(self._start_journey_array)
        sort_idx = np.lexsort((rows, cols)) # sort by vehicle and time 
        rows_sorted = rows[sort_idx]
        cols_sorted = cols[sort_idx]

        # Predefine empty logbook DataFrame
        logbook_df = LogbookSchema.example(size=0)
        logbook_df["id_vehicle"] = cols_sorted + 1 # vehicle IDs start at 1
        logbook_df["dep_dt"] = self._dt_array[rows_sorted]
        logbook_df["arr_dt"] = self._dt_array[rows_sorted] + pd.to_timedelta(self._duration_array[rows_sorted, cols_sorted], unit="h")
        logbook_df["dep_loc"] = self._location_array[rows_sorted-1, cols_sorted]
        step_end_journey = rows_sorted +(self._duration_array[rows_sorted, cols_sorted] / self.model_params.info.temp_res).round().astype(int)
        logbook_df["arr_loc"] = self._location_array[step_end_journey, cols_sorted]
        logbook_df["distance"] = self._distance_array[rows_sorted, cols_sorted]

        # remove id_journey column (will be reindexed later)
        logbook_df = logbook_df.drop(columns=["id_journey"])

        # Create vehicle DataFrame
        first_day = self._dt_array[0].floor('D')
        last_day = self._dt_array[-1].floor('D')
        vehicle_df = pd.DataFrame({
			"id_vehicle": range(1, self._number_vehicles + 1),
			"first_day": [first_day] * self._number_vehicles,
			"last_day":  [last_day] * self._number_vehicles,
			"cluster":   [1] * self._number_vehicles,
            "first_loc": self._location_array[0, :],
		})

        mob_profiles = MobData(
            input_logbook_df=logbook_df,
            input_vehicle_df=vehicle_df
        )

        return mob_profiles