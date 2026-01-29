import pandas as pd
import numpy as np
from typing import Literal, Tuple, Optional
import pandera.pandas as pa
from pandera.typing import Series
import logging
from pydantic import validate_call, ConfigDict

# Configure logger for this module
logger = logging.getLogger(__name__)
		

class LogbooksSchema(pa.DataFrameModel):
	"""Pandera schema for Logbook Dataframe validation."""
	
	id_journey: int = pa.Field(ge=1, coerce=True)
	id_vehicle: int = pa.Field(ge=1, coerce=True)
	dep_dt: pa.DateTime = pa.Field(coerce=True)
	arr_dt: pa.DateTime = pa.Field(coerce=True)
	dep_loc: int = pa.Field(ge=1, coerce=True)
	arr_loc: int = pa.Field(ge=1, coerce=True)
	distance: float = pa.Field(gt=0)
	
	class Config:
		strict = "filter" # remove extra columns
		coerce = True # enforce dtypes
		ordered = False # don't enforce column order

	# check that dep_dt is before arr_dt
	@pa.dataframe_check(
		error="Departure time (dep_dt) must be before arrival time (arr_dt) for all journeys.",
		groupby=None
	)
	def check_time_order(cls, df: pd.DataFrame) -> Series[bool]:
		"""Ensure dep_dt is before arr_dt for all journeys."""
		return df["dep_dt"] < df["arr_dt"]
	
	# check no overlapping journeys per vehicle
	@pa.dataframe_check(
		error="Journeys for the same vehicle cannot overlap. dep_dt must be >= previous arr_dt.",
		groupby=None
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

class Logbooks:
	"""
	Class representing a logbook for vehicle journeys.
	"""

	def __init__(self,  input_df: pd.DataFrame = None, frozen: bool = False):
		"""
		Initialize a Logbook object.
		
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
			If True, the Logbook instance is immutable after creation. Default is False.
		"""
		self._df = None
		self._frozen = frozen
		self._temp_res = None # temporal resolution in hours
		if input_df is not None:
			self.df = input_df 
	
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
	
	@property
	def number_journeys(self) -> int:
		"""Return the number of journeys in the logbook."""
		return len(self._df) if self._df is not None else 0
	
	@property
	def number_vehicles(self) -> int:
		"""Return the number of vehicles in the logbook."""
		if self._df is not None and not self._df.empty:
			return self._df["id_vehicle"].nunique()
		return 0
	
	@property
	def df(self) -> pd.DataFrame:
		"""Get a copy of the logbook DataFrame with calculated duration and speed columns."""
		# Return empty df if self._df is empty
		if self._df is None or self._df.empty:	
			empty_df = LogbooksSchema.example(size=0)
			empty_df['duration'] = pd.Series(dtype='float64')
			empty_df['speed'] = pd.Series(dtype='float64')
			return empty_df
			
		# Calculate duration and speed
		duration = (self._df["arr_dt"] - self._df["dep_dt"]).dt.total_seconds() / 3600  # in hours
		speed = self._df["distance"] / duration  # in km/h
		return self._df.copy().assign(duration=duration, speed=speed)

	@df.setter
	def df(self, value: pd.DataFrame):
		"""Set logbook DataFrame with validation (replaces existing data)."""

		# check if frozen
		self._check_frozen()

		# set the internal DataFrame
		self._df = self._prep_input_df(value)

		# restore location continuity after setting new dataframe
		self.restore_location_continuity()

	def add_journeys(self, input_df: pd.DataFrame) -> None:
		"""
		Add journeys from a DataFrame to the logbook.
		
		Parameters
		----------
		input_df : pd.DataFrame
			DataFrame with journey data to add.
		"""
		# check if frozen
		self._check_frozen()

		# Prepare input DataFrame
		prepared_df = self._prep_input_df(input_df)
		
		# Generate id_journey for new journeys
		prepared_df["id_journey"] = prepared_df["id_journey"] + self.number_journeys
		
		# Append to existing DataFrame
		self._df = pd.concat([self._df, prepared_df], ignore_index=True)

		# Sort by id_vehicle and dep_dt
		self._df = self._df.sort_values(by=["id_vehicle", "dep_dt"]).reset_index(drop=True)

		# Validate DataFrame after insert using Pandera schema
		try:
			LogbooksSchema.validate(self._df)
		except Exception as e:
			# remove added journeys if validation fails
			self.delete_journeys(id_journey=prepared_df["id_journey"].tolist(), reindex=False)
			message = f"Adding journeys failed due to validation error. Error: {str(e)}"
			logger.error(message)
			raise ValueError(message)
		
		# restore location continuity after update
		self.restore_location_continuity()

	def update_journeys(self, input_df: pd.DataFrame) -> None:
		"""
		Update existing journeys in the logbook based on id_journey.
		
		Parameters
		----------
		input_df : pd.DataFrame
			DataFrame with journey data to update.
			Must include 'id_journey' column.
		"""
		# check if frozen
		self._check_frozen()

		# validate input_df
		LogbooksSchema.validate(input_df)

		if self._df is None or self._df.empty:
			message = "Logbook is empty. Cannot update journeys of Logbook."
			logger.error(message)
			raise ValueError(message)
		
		# Backup input_dataframe
		backup_input_df = input_df.copy()
		
		# Update journeys: set index to id_journey for efficient lookup
		self._df.set_index("id_journey", inplace=True)
		input_df.set_index("id_journey", inplace=True)
		
		# Update all columns for journeys that exist in input_df_extended
		self._df.update(input_df)
		
		self._df.reset_index(inplace=True)

		# Sort by id_vehicle and dep_dt
		input_df = input_df.sort_values(by=["id_vehicle", "dep_dt"]).reset_index(drop=True)
		
		# Validate the updated DataFrame
		try:
			LogbooksSchema.validate(self._df)
		except Exception as e:
			# Rollback only affected rows if validation fails
			backup_input_df.set_index("id_journey", inplace=True)
			self._df.set_index("id_journey", inplace=True)
			self._df.update(backup_input_df)
			self._df.reset_index(inplace=True)
			message = f"Update journeys failed due to validation error: {str(e)}"
			logger.error(message)
			raise ValueError(message)
		
		# restore location continuity after update
		self.restore_location_continuity()

	@validate_call
	def delete_journeys(self, id_journey: list, reindex: bool = True) -> None:
		"""Delete journeys by journey ID.

		Parameters
		----------
		id_journey : list[int]
			List of journey IDs to delete.
		reindex : bool, optional
			If True (default), renumber ids after deletion.
		"""
		if self._df is None or self._df.empty:
			return

		# check if frozen
		self._check_frozen()

		# Build deletion mask
		mask = self._df["id_journey"].isin(id_journey)
		self._df = self._df.loc[~mask].copy().reset_index(drop=True)
		
		# Restore location continuity after deletion
		self.restore_location_continuity()

	def _delete_vehicles(self, id_vehicle: list, reindex: bool = True) -> None:
		"""Delete all journeys of specific vehicles.

		Parameters
		----------
		id : list[int]
			List of vehicle IDs whose journeys should be deleted.
		reindex : bool, optional
			If True (default), renumber ids after deletion.
		"""
		if self._df is None or self._df.empty:
			return
		
		# check if frozen
		self._check_frozen()

		# Build deletion mask
		mask = self._df["id_vehicle"].isin(id_vehicle)
		self._df = self._df.loc[~mask].copy().reset_index(drop=True)
		
		if reindex:
			self.reindexing(type="all")

	def _check_frozen(self) -> bool:
		"""Check if the Logbook instance is frozen (immutable)."""
		if self._frozen == True:
			message = "This Logbook instance is frozen and cannot be modified."
			logger.error(message)
			raise AttributeError(message)
	
	@validate_call
	def reindexing(self, type:Literal["all", "id_journey", "id_vehicle"] = "all") -> None:
		"""
		Reindex id_vehicle and/or id_journey columns.
		
		- id_journey: Renumbered from 1 to number_journeys
		- id_vehicle: Renumbered from 1 to number_vehicles
		
		This is useful after adding or removing journeys to maintain continuous IDs.

		Parameters
		----------
		type : str, optional
			Type of reindexing to perform. Options are "all" (default), "id_journey", "id_vehicle".
		"""
		if self._df is None or self._df.empty:
			return
		
		# check if frozen
		self._check_frozen()

		if type not in ["all", "id_journey", "id_vehicle"]:
			message = f"Invalid reindexing type {type}. Must be one of 'all', 'id_journey', 'id_vehicle'."
			logger.error(message)
			raise ValueError(message)
		
		# Sort by id_vehicle and dep_dt first
		self._df = self._df.sort_values(by=["id_vehicle", "dep_dt"]).reset_index(drop=True)
		
		# Reindex id_journey from 1 to number_journeys
		if type in ["all", "id_journey"]:
			self._df["id_journey"] = range(1, self.number_journeys + 1)
		
		# Reindex id_vehicle: map unique vehicles to 1, 2, 3, ...
		if type in ["all", "id_vehicle"]:
			self._df["id_vehicle"] = pd.factorize(self._df["id_vehicle"])[0] + 1
	
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

		# set agregated df as logbook df
		self.df = agg_df

class VehiclesSchema(pa.DataFrameModel):
	"""Pandera schema for Vehicle Dataframe validation."""
	
	id_vehicle: int = pa.Field(ge=1, coerce=True)
	first_day: pa.DateTime = pa.Field(coerce=True)
	last_day: pa.DateTime = pa.Field(coerce=True)
	cluster: int = pa.Field(ge=0, coerce=True,  default=1)
	first_loc: Series[pd.Int64Dtype] = pa.Field(ge=1, nullable=True, coerce=True, default=None)
	
	class Config:
		strict = "filter" # remove extra columns
		coerce = True # enforce dtypes
		ordered = False # don't enforce column order
		add_missing_columns=True

	# check that dep_dt is before arr_dt
	@pa.dataframe_check(
		error="First day (first_day) must be before last day (last_day) for all vehicles.",
		groupby=None
	)
	def check_time_order(cls, df: pd.DataFrame) -> Series[bool]:
		"""Ensure first_day is before last_day for all vehicles."""
		return df["first_day"] <= df["last_day"]
	
	# check id_vehicle is unique
	@pa.dataframe_check(
		error="id_vehicle must be unique. No duplicate vehicle IDs allowed.",
		groupby=None
	)
	def check_id_vehicle_unique(cls, df: pd.DataFrame) -> Series[bool]:
		"""Ensure id_vehicle is unique across all rows."""
		return ~df["id_vehicle"].duplicated(keep=False)

class Vehicles:
	"""
	Class representing vehicles with aggregated statistics from journeys.
	"""

	def __init__(self, input_df: pd.DataFrame = None, frozen: bool = False):
		"""
		Initialize a Vehicle object.
		
		Parameters
		----------
		input_df : pd.DataFrame, optional
			Initial DataFrame with vehicle data.
			Expected columns and dtypes:
			- id_vehicle: int
			- first_day: datetime64[D]
			- last_day:  datetime64[D]
			- cluster:   int (optional: default 1)
			- first_loc: int (optional: Default None)
		frozen : bool, optional
			If True, the Vehicle instance is immutable after creation. Default is False.
		"""
		self._df = None
		self._frozen = frozen
		if input_df is not None:
			self.df = input_df
	
	@property
	def df(self) -> pd.DataFrame:
		"""Get a copy of the vehicle DataFrame."""
		if self._df is None or self._df.empty:
			return VehiclesSchema.example(size=0)

		return self._df.copy()
	
	@df.setter
	def df(self, value: pd.DataFrame):
		"""Set vehicle DataFrame with validation (replaces existing data)."""
		# check if frozen
		self._check_frozen()
		# set the internal DataFrame
		self._df = VehiclesSchema.validate(value)
	
	def add_vehicles(self, input_df: pd.DataFrame) -> None:
		"""
		Add vehicles from a DataFrame.
		
		Parameters
		----------
		input_df : pd.DataFrame
			DataFrame with vehicle data to add.
		"""
		# check if frozen
		self._check_frozen()

		# Validate input DataFrame
		new_vehicles_df = VehiclesSchema.validate(input_df)
		
		# Append to existing DataFrame
		new_df = pd.concat([self._df, new_vehicles_df], ignore_index=True)
		
		# Validate combined DataFrame
		self._df = VehiclesSchema.validate(new_df)

		# set labels for clusters
		self.set_label_clusters()
	
	
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
			- cluster:   int
			- first_loc: int (optional)
		"""
		# check if frozen
		self._check_frozen()

		# Validate input DataFrame
		update_df = VehiclesSchema.validate(input_df)
		
		if self._df is None or self._df.empty:
			message = "Vehicle DataFrame is empty. Cannot update vehicles."
			logger.error(message)
			raise ValueError(message)
		
		# Set index to id_vehicle for efficient lookup
		self._df.set_index("id_vehicle", inplace=True)
		update_df.set_index("id_vehicle", inplace=True)
		
		# Update all columns for vehicles that exist in update_df
		self._df.update(update_df)
		
		self._df.reset_index(inplace=True)

		# set labels for clusters
		self.set_label_clusters()
	
	
	def _check_frozen(self) -> bool:
		"""Check if the Vehicle instance is frozen (immutable)."""
		if self._frozen == True:
			message = "This Vehicle instance is frozen and cannot be modified."
			logger.error(message)
			raise AttributeError(message)
	
	def _delete_vehicles(self, id_vehicle: list) -> None:
		"""Delete vehicles by vehicle ID.
		
		Parameters
		----------
		id_vehicle : list[int]
			List of vehicle IDs to delete.
		"""
		# check if frozen
		self._check_frozen()

		if self._df is None or self._df.empty:
			return
		
		# Build deletion mask
		mask = self._df["id_vehicle"].isin(id_vehicle)
		self._df = self._df.loc[~mask].copy().reset_index(drop=True)

		# set labels for clusters
		self.set_label_clusters()

	def generate_vehicles_from_logbook(self, logbook: Logbooks) -> None:
		"""
		Generate vehicle DataFrame from a Logbook instance.
		
		Parameters
		----------
		logbook : Logbook
			Logbook instance with journey data to generate vehicles from.
		"""
		if isinstance(logbook, Logbooks) == False:
			message = "logbook must be an instance of Logbook class."
			logger.error(message)
			raise TypeError(message)
		
		logbook_df = logbook.df
		
		if logbook_df is None or logbook_df.empty:
			self._df = VehiclesSchema.example(size=0)
			return
		
		# Group by id_vehicle to get first_day, last_day and first_location
		grouped = logbook_df.groupby("id_vehicle").agg(
			first_day=pd.NamedAgg(column="dep_dt", aggfunc="min"),
			last_day=pd.NamedAgg(column="arr_dt", aggfunc="max"),
			first_loc=pd.NamedAgg(column="dep_loc", aggfunc="first")
		).reset_index()
		
		# Convert to date only
		grouped["first_day"] = grouped["first_day"].dt.floor('D')
		grouped["last_day"] = grouped["last_day"].dt.floor('D')
		
		# Assign cluster as 1 for all vehicles (placeholder)
		grouped["cluster"] = 1
		
		self._df = VehiclesSchema.validate(grouped)

		# set labels for clusters
		self.set_label_clusters()
	
	def set_first_loc_from_logbook(self, logbook: Logbooks) -> None:
		"""
		Set first_loc for each vehicle based on the first dep_loc in the logbook.
        
		Parameters
		----------
		logbook : Logbook
			Logbook instance with journey data to extract first locations from.
		"""
		if isinstance(logbook, Logbooks) == False:
			message = "logbook must be an instance of Logbook class."
			logger.error(message)
			raise TypeError(message)
		
		logbook_df = logbook.df
        
		if logbook_df is None or logbook_df.empty:
			return
        
		# Get first dep_loc per vehicle
		first_loc = logbook_df.sort_values(by=['dep_dt']).groupby('id_vehicle').first().reset_index()
		first_loc = first_loc[['id_vehicle', 'dep_loc']].rename(columns={'dep_loc': 'first_loc'})
        
		# Remove existing first_loc column if present to avoid _x/_y suffix
		if 'first_loc' in self._df.columns:
			self._df = self._df.drop(columns=['first_loc'])
        
		# Merge into vehicle DataFrame
		self._df = self._df.merge(
			first_loc,
			on='id_vehicle',
			how='left'
		)

		# set first_loc of non driving vehicles to 1: nan --> 1
		self._df.loc[self._df["first_loc"].isna(), "first_loc"] = 1

	@validate_call
	def set_label_clusters(self, labels: list[str] | None = None, clusters: list[int] | None = None) -> None:
		"""Set labels for vehicle clusters.
		Parameters
		----------
		labels : list of str
			Label or list of labels to assign to clusters.
		clusters : list of int
			Cluster or list of clusters to assign labels to.
		"""
		# check if frozen
		self._check_frozen()

		# Get unique cluster values
		cluster_values = self._df["cluster"].unique().tolist()
		# Assign default clusters if none provided
		if labels is None:
			self.labels_clusters = [f"cluster={c}" for c in cluster_values]
			return
		
		# Check dimensions of lists
		if clusters is not None:
			if len(clusters) != len(labels):
				message = "Length of clusters and labels must match."
				logger.error(message)
				raise ValueError(message)
		# Check number of clusters in vehicle dataframe
		number_of_clusters = len(cluster_values)
		if number_of_clusters != len(labels):
			message = f"Number of unique clusters in vehicle DataFrame ({number_of_clusters}) does not match number of provided labels ({len(labels)})."
			logger.error(message)
			raise ValueError(message)
		
		# Define labels for clusters
		self.labels_clusters = labels

class ClustersSchema(pa.DataFrameModel):
	"""Pandera schema for Logbook Dataframe validation."""
	id_cluster: int = pa.Field(ge=1, coerce=True)
	label: int = pa.Field(ge=1, coerce=True)
class Clusters:
	"Class representing clusters of vehicles."
	def __init__(self, vehicles: Vehicles | None = None):
		"""
		Generate clusters DataFrame.

		Parameters
		----------
		vehicles : Vehicle, optional
			Vehicle instance with vehicle data including 'id_cluster' column.
		"""
		if vehicles is None:
			# Initialize empty clusters DataFrame
			self._df =ClustersSchema.example(size=0)
		elif isinstance(vehicles, Vehicles):
			# Generate clusters from vehicles
			self.update_cluster_from_vehicles(vehicles)
		else:
			message = "vehicles must be an instance of Vehicle class."
			logger.error(message)
			raise TypeError(message)

	@property
	def df(self) -> pd.DataFrame:
		"""Get the Dataframe of clusters."""
		return self._df.copy()
	
	def update_cluster_from_vehicles(self, vehicles: Vehicles) -> None:
		"""
		Update clusters DataFrame based on current vehicle DataFrame.

		Parameters
		----------
		vehicles : Vehicle
			Vehicle instance with vehicle data including 'id_cluster' column.
		"""
		# Validate vehicle DataFrame
		vehicles_df = vehicles.df

		cluster_ids = vehicles_df["id_cluster"].unique()
		cluster_labels = [f"Cluster {cid}" for cid in cluster_ids]
		update_df = pd.DataFrame({
			"id_cluster": cluster_ids,
			"label": cluster_labels
		})
		update_df = ClustersSchema.validate(update_df)
		if self._df is None or self._df.empty:
			self._df = update_df
			return
		
		existing_df = self.df

		# Set index to id_cluster for efficient lookup
		existing_df.set_index("id_cluster", inplace=True)
		update_df.set_index("id_cluster", inplace=True)
		
		# Update all columns for clusters that exist in update_df
		update_df.update(existing_df)

		update_df.reset_index(inplace=True)

		self._df = ClustersSchema.validate(update_df)
	
	def update_clusters_from_df(self, input_df: pd.DataFrame) -> None:
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
		# Validate input DataFrame
		update_df = ClustersSchema.validate(input_df)
		existing_df = self.df
		
		if self._df is None or self._df.empty:
			message = "Clusters DataFrame is empty. Cannot update clusters."
			logger.error(message)
			raise ValueError(message)
		
		# Set index to id_cluster for efficient lookup
		existing_df.set_index("id_cluster", inplace=True)
		update_df.set_index("id_cluster", inplace=True)
		
		# Update all columns for clusters that exist in update_df
		existing_df.update(update_df)
		existing_df.reset_index(inplace=True)

		self._df = ClustersSchema.validate(existing_df)
		
class MobData:
	"""
	Base class for mobility data in the champpy framework.
	"""
	def __init__(
		self,
		input_logbook_df: pd.DataFrame,
		input_vehicle_df: pd.DataFrame | None = None,
		frozen: bool = False
		):
		"""
		Initialize a MobData object.

		Parameters
		----------
		input_logbook_df : pd.DataFrame
			Input DataFrame for the logbook.
			Expected columns and dtypes:
			- id_vehicle: int
			- dep_dt: datetime64[ns]
			- arr_dt: datetime64[ns]
			- dep_loc: str
			- arr_loc: str
			- distance: float
		input_vehicle_df : pd.DataFrame, optional
			Input DataFrame for the vehicles.
			Expected columns and dtypes:
			- id_vehicle: int
			- first_day: datetime64[D]
			- last_day:  datetime64[D]
			- cluster:   int
			- first_loc: int (optional)
		frozen : bool, optional
			If True, the MobData instance is immutable after creation. Default is False.
		"""
		self.logbook = Logbooks(input_df=input_logbook_df, frozen=frozen)
		if input_vehicle_df is not None:
			self.vehicles = Vehicles(input_df=input_vehicle_df)
			if self.vehicles.df["first_loc"].isnull().all():
				self.vehicles.set_first_loc_from_logbook(self.logbook)
		else:
			self.vehicles = Vehicles(frozen=False)
			self.vehicles.generate_vehicles_from_logbook(self.logbook)

		# set frozen after initialization
		self.vehicles._frozen = frozen  
		self._cleaned = False
	
	def __copy__(self):
		return MobData(self.logbook.df, self.vehicles.df)
	
	def add_mob_data(self, input_mob_data: "MobData") -> None:
		"""
		Add mobility data from another MobData instance.

		Parameters
		----------
		other : MobData
			Another MobData instance to add data from.
		"""
		if not isinstance(input_mob_data, MobData):
			message = "other must be an instance of MobData."
			logger.error(message)
			raise TypeError(message)
		
		# extract dataframes
		logbook_df = input_mob_data.logbook.df
		vehicles_df = input_mob_data.vehicles.df

		# Make sure vehicle IDs and clusters are unique across both datasets
		max_id_vehicle = self.vehicles.df["id_vehicle"].max() if not self.vehicles.df.empty else 0
		max_cluster = self.vehicles.df["cluster"].max() if not self.vehicles.df.empty else 0
		vehicles_df["id_vehicle"] += max_id_vehicle + 1 
		logbook_df["id_vehicle"] += max_id_vehicle + 1
		vehicles_df["cluster"] += max_cluster + 1 
		
		# Add to logbook and vehicles
		self.vehicles.add_vehicles(vehicles_df)
		self.logbook.add_journeys(logbook_df)

		# Reindex IDs id_vehicles and id_journey after addition
		self.reindexing()

	def reindexing(self) -> None:
		"""
		Reindex vehicle and journey IDs in the MobData instance.
		- id_vehicle: Renumbered from 1 to number_vehicles
		- id_journey: Renumbered from 1 to number_journeys
		"""
		# Reindex vehicles based on logbook starting from 1
		unique_vehicles = self.logbook.df["id_vehicle"].unique()
		reindex_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_vehicles), start=1)}
		self.logbook._df["id_vehicle"] = self.logbook._df["id_vehicle"].map(reindex_map)
		self.vehicles._df["id_vehicle"] = self.vehicles._df["id_vehicle"].map(reindex_map)

		# Reindex cluster starting from 0
		unique_clusters = self.vehicles.df["cluster"].unique()
		cluster_map = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(sorted(unique_clusters))}
		self.vehicles._df["cluster"] = self.vehicles._df["cluster"].map(cluster_map)

		# Reindex id_journey
		self.logbook.reindexing(type="id_journey")

	@validate_call
	def delete_vehicles(self, id_vehicle: list, reindex: bool = True) -> None:
		"""Delete vehicles from logbook and vehicle tables.

		Parameters
		----------
		id_vehicle : list[int]
			List of vehicle IDs to delete.
		reindex : bool, optional
			If True, reindex IDs in the logbook after deletion. Default True.
		"""
		# Remove journeys for these vehicles
		self.logbook._delete_vehicles(id_vehicle=id_vehicle, reindex=reindex)
		# Remove corresponding vehicle rows
		self.vehicles._delete_vehicles(id_vehicle=id_vehicle)

	@validate_call
	def get_annual_km(self, aggregate: bool = True, agg_method: Literal["mean", "sum"] = "mean") -> pd.DataFrame:
		"""
		Calculate annual kilometrage per vehicle.

		Params
		------
		aggregate : bool, optional
			If True, return aggregated annual kilometers. Default is True.
		agg_method : str, optional
			Method to aggregate annual kilometers. Options are "mean" or "sum". Default is "mean".

		Returns
		-------
		pd.DataFrame
			DataFrame with columns 'id_vehicle' and 'annual_km'.
		"""
		if agg_method not in ["mean", "sum"]:
			message = "agg_method must be either 'mean' or 'sum'."
			logger.error(message)
			raise ValueError(message)
		
		# Empty guard
		if self.logbook.df is None or self.logbook.df.empty:
			return pd.DataFrame(columns=["id_vehicle", "annual_km"])

		# Calculate total distance per vehicle
		distance_per_vehicle = self.logbook.df.groupby("id_vehicle")["distance"].sum().reset_index()

		# Calculate number of active days per vehicle
		number_days = (self.vehicles.df["last_day"] - self.vehicles.df["first_day"] + pd.Timedelta(days=1)).dt.days

		# Merge to ensure all vehicles are included
		df = self.vehicles.df[["id_vehicle"]].copy()
		df = df.merge(distance_per_vehicle, on="id_vehicle", how="left")
		df["distance"] = df["distance"].fillna(0)
		df["number_days"] = number_days.values
		df["annual_km"] = df["distance"] / df["number_days"] * 365
		df["annual_km"] = df["annual_km"].fillna(0)
		
		if aggregate:
			if agg_method == "mean":
				annual_km = df["annual_km"].mean()
			else:  # agg_method == "sum"
				annual_km = df["annual_km"].sum()
			output_df = pd.DataFrame({"annual_km": [annual_km]})
		else:
			output_df = df[["id_vehicle", "annual_km"]]
		return output_df
	
	@validate_call
	def get_share_of_time_at_locations(self, aggregate: bool = True) -> pd.DataFrame:
		"""
		Calculate the share of time the vehicles spend at each location.
		Params
		------
		aggregate : bool, optional
			If True, return aggregated share per location across all vehicles.
			If False, return share per vehicle and location. Default is True.

		Returns
		-------
		pd.DataFrame
			DataFrame with columns 'id_vehicle', 'location', and 'share'.
		"""
		# Get total hours per vehicle
		total_day_per_vehicle = self.vehicles.df["last_day"] - self.vehicles.df["first_day"] + pd.Timedelta(days=1)
		total_hours_per_vehicle = total_day_per_vehicle.dt.total_seconds() / 3600

		# Get extended mobility data and group by location and id_vehicle and sum duration
		extended_mob_data = MobDataExtended(self, splitdays=False).df

		# Aggregate
		if aggregate:
			# group by location and sum duration
			location_duration_df = (
				extended_mob_data
				.groupby(["location"])["duration"]
				.sum()
				.reset_index()
			)

			# add new column total_hours with total hours of all vehicles
			location_duration_df["total_hours"] = total_hours_per_vehicle.sum()

		else:
			# group by location and id_vehicle and sum duration
			location_duration_df = (
				extended_mob_data
				.groupby(["location", "id_vehicle"])["duration"]
				.sum()
				.reset_index()
			)

			# merge total hours per vehicle
			location_duration_df = location_duration_df.merge(
				total_hours_per_vehicle.rename("total_hours"),
				left_on="id_vehicle",
				right_index=True,
				how="left"
			)

		# calculate share per vehicle at each location
		location_duration_df["share"] = location_duration_df["duration"] / location_duration_df["total_hours"]
		# replace nan with 0
		location_duration_df["total_hours"] = location_duration_df["total_hours"].fillna(0)
		location_duration_df["share"] = location_duration_df["share"].fillna(0)
		# rename duration to hours_at_location
		location_duration_df = location_duration_df.rename(columns={"duration": "hours_at_location"})

		return location_duration_df

class MobProfiles(MobData):
	""" 
	Class for modeled mobility profiles. Child of MobData but first_day and last_day of all vehicles are equal. 
	MobProfiles are immutable after creation. 

	Parameters
	----------
	first_day : pd.Timestamp
		First day of all vehicles in the profile.
	last_day : pd.Timestamp
		Last day of all vehicles in the profile.
	number_vehicles : int
		Number of vehicles in the profile.
	input_logbook_df : pd.DataFrame
		Input DataFrame for the logbook.
	"""
	def __init__(
		self,
		first_day: pd.Timestamp,
		last_day:  pd.Timestamp,
		number_vehicles: int,
		input_logbook_df: pd.DataFrame,
	):	
		# Create vehicle dataframe with same first_day and last_day for all vehicles
		vehicle_df = pd.DataFrame({
			"id_vehicle": range(1, number_vehicles + 1),
			"first_day": [first_day.floor('D')] * number_vehicles,
			"last_day":  [last_day.floor('D')] * number_vehicles,
			"cluster":   [1] * number_vehicles
		})
		super().__init__(input_logbook_df=input_logbook_df, input_vehicle_df=vehicle_df)
		self.frozen = True  # make MobProfiles immutable after creation

class MobDataExtended:
	"""
	Extended MobData with additional attributes for modeling.

	Parameters
	----------
	mob_data : MobData
		Base MobData instance.
	"""
	def __init__(self, mob_data: MobData, splitdays: bool = True):
		
		if not isinstance(mob_data, MobData):
			message = "mob_data must be an instance of MobData class."
			logger.error(message)
			raise TypeError(message)
		
		# Predefine empty DataFrame with required columns
		self._df = pd.DataFrame({
			'id_vehicle': pd.Series(dtype='int64'),
			'start_dt': pd.Series(dtype='datetime64[ns]'),
			'end_dt': pd.Series(dtype='datetime64[ns]'),
			'location': pd.Series(dtype='int64'),
			'speed': pd.Series(dtype='float64')
		})

		# Extend mob_data to include standing and non-driving vehicles
		self._extended_mob_data(mob_data)
		
		# Split multi-day rows if required
		self._split_multi_day_rows(splitdays=splitdays)

		# Join the 'cluster' column from t_vehicle into t_location
		self._df = self._df.merge(mob_data.vehicles._df[['id_vehicle', 'cluster']], on='id_vehicle', how='left')
		self._df['cluster'] = self._df['cluster'].astype('int64')

		self.labels_clusters = mob_data.vehicles.labels_clusters

	@property
	def df(self) -> pd.DataFrame:
		"""Get a copy of the extended MobData DataFrame."""			
		# Calculate distance and duration
		duration = (self._df["end_dt"] - self._df["start_dt"]).dt.total_seconds() / 3600  # in hours
		distance = self._df["speed"] * duration  # in km/h
		return self._df.copy().assign(duration=duration, distance=distance)

	def _extended_mob_data(self, mob_data: MobData):
		"""
		Create extended DataFrame with additional attributes.

		Returns
		-------
		pd.DataFrame
			Extended DataFrame.
		"""
		# convert automatically to uniform temporal resolution
		if mob_data.logbook.temp_res is None:
			# find the minimum temporal resolution in hours
			min_res = mob_data.logbook.df.apply(
				lambda row: (row['arr_dt'] - row['dep_dt']).total_seconds() / 3600, axis=1
			).min()
			mob_data.logbook.temp_res = min_res

		lb_df = mob_data.logbook._df
		vehicles_df = mob_data.vehicles._df

		# determine first_loc for vehicles if nan
		if any(vehicles_df["first_loc"].isna()):
			vehicles_df.set_first_loc_from_logbook(mob_data.logbook)

		# Identify non-drivers
		mask_nondriver_vehicle = ~vehicles_df['id_vehicle'].isin(lb_df['id_vehicle'])
		n_nondriver_vehicle = mask_nondriver_vehicle.sum()
		
		# Create t_nondriver only if there are non-driver vehicles
		if n_nondriver_vehicle > 0:
			# Use first_loc if available, otherwise use default location 1
			nondriver_locations = vehicles_df.loc[mask_nondriver_vehicle, 'first_loc'].astype('int64')
			
			nondriver_df = pd.DataFrame({
				'id_vehicle': vehicles_df.loc[mask_nondriver_vehicle, 'id_vehicle'],
				'start_dt': vehicles_df.loc[mask_nondriver_vehicle, 'first_day'],
				'end_dt': vehicles_df.loc[mask_nondriver_vehicle, 'last_day'],
				'location': nondriver_locations,
				'speed': 0
			})
		else:
			nondriver_df = pd.DataFrame()
		
		# return if all vehicles are non-drivers
		if n_nondriver_vehicle == len(vehicles_df):
			self._df = nondriver_df.sort_values(by=['id_vehicle', 'start_dt']).reset_index(drop=True)
			return
		
		# Filter vehicles with journeys
		vehicle_df_drivers = vehicles_df.loc[~mask_nondriver_vehicle]

		# Find first and last track of each vehicle
		group = lb_df.groupby('id_vehicle')
		first_id_track = group['id_journey'].min()
		last_id_track = group['id_journey'].max()

		# Define rows for locations before the first trip
		start_df = pd.DataFrame({
			'id_vehicle': vehicle_df_drivers['id_vehicle'],
			'start_dt': vehicle_df_drivers['first_day'],
			'end_dt': lb_df.dep_dt[lb_df['id_journey'].isin(first_id_track)].values,
			'location': lb_df.dep_loc[lb_df['id_journey'].isin(first_id_track)].values,
			'speed': 0
		})

		# Define rows for locations after the last trip
		end_df = pd.DataFrame({
			'id_vehicle': vehicle_df_drivers['id_vehicle'],
			'start_dt': lb_df.arr_dt[lb_df['id_journey'].isin(last_id_track)].values,
			'end_dt': vehicle_df_drivers['last_day'] + pd.Timedelta(days=1),
			'location': lb_df.arr_loc[lb_df['id_journey'].isin(last_id_track)].values,
			'speed': 0
		})

		# Define rows for locations between trips
		standing_df = pd.DataFrame({
			'id_vehicle': lb_df.id_vehicle[~lb_df['id_journey'].isin(last_id_track)].values,
			'start_dt': lb_df.arr_dt[~lb_df['id_journey'].isin(last_id_track)].values,
			'end_dt': lb_df.dep_dt[~lb_df['id_journey'].isin(first_id_track)].values,
			'location': lb_df.arr_loc[~lb_df['id_journey'].isin(last_id_track)].values,
			'speed': 0
		})

		# Define rows for location driving
		driving_df = pd.DataFrame({
			'id_vehicle': lb_df['id_vehicle'],
			'start_dt': lb_df['dep_dt'],
			'end_dt': lb_df['arr_dt'],
			'location': 0,
			'speed': lb_df['distance'] / ((lb_df['arr_dt'] - lb_df['dep_dt']).dt.total_seconds() / 3600)
		})

		# Merge dataframes
		self._df = pd.concat([nondriver_df, start_df, standing_df, driving_df, end_df]).sort_values(by=['id_vehicle', 'start_dt'])
		self._df.reset_index(drop=True, inplace=True)

	def _split_multi_day_rows(self, splitdays: bool) -> pd.DataFrame:
		"""
		Split multi-day rows in t_location into single-day rows.
		"""
		if not splitdays:
			return

		# Split multi-day rows: vehicle is at one location over several days
		day_start = self._df['start_dt'].dt.floor('D')
		day_end = self._df['end_dt'].dt.floor('D')
		n_days = (day_end - day_start).dt.days + 1
		row_end_at_midnight = (self._df['end_dt'].dt.time == pd.Timestamp('00:00:00').time()) & (n_days > 1)

		# determine days per vehicle
		group = self._df.groupby('id_vehicle')
		first_day = group['start_dt'].min()
		last_day = group['end_dt'].max()
		days_per_vehicle = (last_day - first_day).dt.days
		
		# Abort if no multi-day rows exist
		if all(n_days == 1) or all(days_per_vehicle == 1):
			return self._df
		
		# New row for the last day of a multi-day row
		log_add_row_end = ~row_end_at_midnight & (n_days > 1)
		split_end_df = pd.DataFrame({
			'id_vehicle': self._df.loc[log_add_row_end, 'id_vehicle'],
			'start_dt': self._df.loc[log_add_row_end, 'end_dt'].dt.floor('D'),
			'end_dt': self._df.loc[log_add_row_end, 'end_dt'],
			'location': self._df.loc[log_add_row_end, 'location'],
			'speed': self._df.loc[log_add_row_end, 'speed']
		})

		# New rows for constant days in the middle: vehicle is at the same location over the whole day
		n_parking_days = n_days - 2
		n_parking_days[n_parking_days < 0] = 0
		parking_start_day = day_end[n_parking_days > 0] - pd.to_timedelta(n_parking_days[n_parking_days > 0], unit='D')
		parking_end_day = day_end[n_parking_days > 0]
		parking_days = [pd.date_range(start, end, inclusive='left') 
							for start, end in zip(parking_start_day, parking_end_day)]
		split_mid_df = pd.DataFrame({
			'id_vehicle': np.repeat(self._df.loc[n_parking_days > 0, 'id_vehicle'].values, n_parking_days[n_parking_days > 0]),
			'start_dt': np.concatenate(parking_days),
			'end_dt': np.concatenate(parking_days) + pd.Timedelta(days=1),
			'location': np.repeat(self._df.loc[n_parking_days > 0, 'location'].values, n_parking_days[n_parking_days > 0]),
			'speed': 0
		})

		# Modify t_location for the first day of multi-day row
		self._df.loc[n_days > 1, 'end_dt'] = day_start[n_days > 1] + pd.Timedelta(days=1)

		# Merge
		self._df = pd.concat([self._df, split_mid_df, split_end_df]).sort_values(by=['id_vehicle', 'start_dt'])

		# reset index
		self._df.reset_index(drop=True, inplace=True)
	
	def extract_cluster(self, cluster_id: int, copy: bool = True) -> "MobDataExtended":
		"""
		Extract a specific cluster from the extended mobility data.

		Parameters
		----------
		cluster_id : int
			Cluster ID to extract.

		Returns
		-------
		MobDataExtended
			New MobDataExtended instance with data for the specified cluster.
		"""
		if copy:
			cluster_df = self._df[self._df['cluster'] == cluster_id].copy().reset_index(drop=True)
		else:
			cluster_df = self._df[self._df['cluster'] == cluster_id].reset_index(drop=True)
		new_instance = MobDataExtended.__new__(MobDataExtended)
		new_instance._df = cluster_df
		return new_instance
