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

from champpy.core.mobility_data import MobProfiles
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


    def generate_mob_profiles(self, user_params: UserParamsMobModel) -> MobProfiles:
        
        # Set random seed
        np.random.seed(user_params.random_seed)

        # Predefine variables
        self._predefine_vars(user_params=user_params)

        # Define a possible location for the first step based on the transition matrix
        possible_start_locations = [1]# TODO : determine based on transition matrix
        self._location_array[0, :] = np.random.choice(possible_start_locations, size=self._number_vehicles, replace=True)
    
        # Add rich progress bar for vehicle loop
        for v in track(range(self._number_vehicles), description="Generating mobility profiles..."):
            for t in range(1, self._number_steps):

                # Determine new location based on transition matrix
                self._generate_location(t, v)
                location_t = self._location_array[t, v]
                location_tminus1 = self._location_array[t-1, v]

                if location_t == 0 and location_tminus1 != 0 and t != self._number_steps - 1:
                    # Start of a new journey
                    self._start_journey_array[t, v] = True
                    index_start = t
                elif (location_t >0 or t == (self._number_steps - 1)) and location_tminus1 == 0: 
                    # End of the current journey
                    journey_duration_h = (t - index_start) * self.model_params.info.temp_res
                    self._duration_array[index_start,v] = journey_duration_h
                    #self._generate_speed(t, v, index_start, journey_duration_h)
                    if t == (self._number_steps - 1):
                        # Intervene if journey ends at the last time step
                        # Set location to most frequent location:
                        most_frequent_location = mode(self._location_array[:, v])[0]
                        self._location_array[t, v] = most_frequent_location


        # Convert arrays to DataFrame
        logbook_df = self._convert_arrays2logbook_df()
        # Create MobProfiles dataclass
        mob_profile = MobProfiles()
    
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

        # split vehicles into clusters
        cluster_array = self._split_vehicles_per_cluster(self._number_vehicles)
    
        # Determine index of parameters based on cluster_array and typedays_array
        self._index_params_array = (
            self.model_params.df.reset_index()
            .pivot(index="typeday", columns="cluster", values="index")
            .loc[typedays_array, cluster_array]
            .to_numpy()
        )
        
        # Initialize arrays for location, speed, distance, duration
        self._location_array = np.zeros((self._number_steps, self._number_vehicles), dtype=int) # location of vehicles
        self._speed_array = np.zeros((self._number_steps, self._number_vehicles)) # speed of journeys
        self._distance_array = np.zeros((self._number_steps, self._number_vehicles)) # distance of journeys
        self._duration_array = np.zeros((self._number_steps, self._number_vehicles)) # duration of journeys
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
    
    def _generate_location(self, t, v) -> None:
        """
        Generate the location of the vehicle based on the transition matrix.
        Args:
            t: Time step index.
            v: Vehicle index.
        """
        # Index of parameters
        index_params = self._index_params_array[t, v]

        # Location of previous step
        location_tminus1 = self._location_array[t-1, v]

        # generate random number to determine new location
        rand1 = np.random.rand()

        # Get transition matrix from params dataframe
        trans_matrix = self.model_params.df.loc[index_params, "transition_matrix"]

        # Extract transition vector from transition matrix for the current day index and previous location
        trans_vector = trans_matrix[self._index_day_array[t],location_tminus1, :]

        # Calculate cumulative transition vector
        cum_trans_vector = np.cumsum(trans_vector)

        # Find the first index where rand1 <= cum_trans_vector. This is the new location.
        location_t = np.searchsorted(cum_trans_vector, rand1, side="right")

        if location_t == 3:
            # Debugging output
            logger.debug(f"At time step {t}, vehicle {v}: rand1={rand1}, cum_trans_vector={cum_trans_vector}, location_t={location_t}")
        
        # Set new location
        self._location_array[t, v] = location_t
    
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

    def _convert_arrays2logbook_df(self) -> pd.DataFrame:
        """
        Convert the generated arrays to a pandas DataFrame representing the logbook.
        Returns:
            pd.DataFrame: DataFrame containing the logbook data.
        """
        # TODO
        pass

