import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
import logging
from typing import Literal
from pydantic import validate_call
from champpy.utils.data_utils import Event

# Basisklasse für Mobility-Komponenten
import pandas as pd

class BaseMobilityComponent:
	_schema = None  # In Subklassen überschreiben

	def __init__(self, input_df: pd.DataFrame = None, frozen: bool = False):
		self._df = None
		self._frozen = frozen
		if input_df is not None:
			self.df = input_df

	@property
	def df(self) -> pd.DataFrame:
		if self._df is None:
			return pd.DataFrame()
		return self._df.copy()

	@df.setter
	def df(self, value: pd.DataFrame):
		self._check_frozen()
		if self._schema is not None:
			self._df = self._schema.validate(value)
		else:
			self._df = value

	@property
	def is_empty(self) -> bool:
		return self._df is None or self._df.empty

	def _check_frozen(self):
		if self._frozen:
			raise AttributeError("This instance is frozen and cannot be modified.")

# Configure logger for this module
logger = logging.getLogger(__name__)


		
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
			message = "Logbooks are empty. Cannot update journeys of Logbooks."
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
			self._reindexing(type="all")

	def _check_frozen(self) -> bool:
		"""Check if the Logbooks instance is frozen (immutable)."""
		if self._frozen == True:
			message = "This Logbooks instance is frozen and cannot be modified."
			logger.error(message)
			raise AttributeError(message)
	
	@validate_call
	def _reindexing(self, type:Literal["all", "id_journey", "id_vehicle"] = "all") -> None:
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
	"""Pandera schema for Vehicles Dataframe validation."""
	
	id_vehicle: int = pa.Field(ge=1, coerce=True)
	first_day: pa.DateTime = pa.Field(coerce=True)
	last_day: pa.DateTime = pa.Field(coerce=True)
	id_cluster: int = pa.Field(ge=1, coerce=True,  default=1)
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
		self._df = None
		self._frozen = frozen
		if input_df is not None:
			self.df = input_df
		self._event_on_logbooks = Event[int]()  # Event triggered on vehicle deletion
		self._event_on_clusters = Event[self]()  # Event triggered on vehicle update
	
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

		# Triggger event to update cluster labels
		self._event_on_add_update.trigger(new_vehicles_df)
	
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
		# check if frozen
		self._check_frozen()

		# Validate input DataFrame
		update_df = VehiclesSchema.validate(input_df)
		
		if self._df is None or self._df.empty:
			message = "VehiclesDataFrame is empty. Cannot update vehicles."
			logger.error(message)
			raise ValueError(message)
		
		# Set index to id_vehicle for efficient lookup
		existing_df = self._df.copy()
		existing_df.set_index("id_vehicle", inplace=True)
		update_df.set_index("id_vehicle", inplace=True)
		
		# Update all columns for vehicles that exist in update_df
		existing_df.update(update_df)
		
		existing_df.reset_index(inplace=True)
		self._df = VehiclesSchema.validate(existing_df)

		# Triggger event to update cluster labels
		self._event_on_clusters.trigger(self)
	
	def delete_vehicles(self, id_vehicle: list) -> None:
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
		grouped = logbooks_df.groupby("id_vehicle").agg(
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

		# Triggger event to update cluster labels
		self._event_on_clusters.trigger(self)
	
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
		first_loc = logbooks_df.sort_values(by=['dep_dt']).groupby('id_vehicle').first().reset_index()
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
		self._df["first_loc"] = self._df["first_loc"].astype("Int64")
	
	def _check_frozen(self) -> bool:
		"""Check if the Vehiclesinstance is frozen (immutable)."""
		if self._frozen == True:
			message = "This Vehiclesinstance is frozen and cannot be modified."
			logger.error(message)
			raise AttributeError(message)
		return False


class ClustersSchema(pa.DataFrameModel):
	"""Pandera schema for Logbooks Dataframe validation."""
	id_cluster: int = pa.Field(ge=1, coerce=True)
	label: str = pa.Field(coerce=True)
	
class Clusters:
	"Class representing clusters of vehicles."
	def __init__(self, vehicles: Vehicles | None = None):
		"""
		Generate clusters DataFrame.

		Parameters
		----------
		vehicles : Vehicles, optional
			Vehicles instance with vehicle data including 'id_cluster' column.
		"""
		if vehicles is None:
			# Initialize empty clusters DataFrame
			self._df =ClustersSchema.example(size=0)
		elif isinstance(vehicles, Vehicles):
			# Generate clusters from vehicles
			self._df = pd.DataFrame()
			self.update_clusters_from_vehicles(vehicles)
		else:
			message = "vehicles must be an instance of Vehicles class."
			logger.error(message)
			raise TypeError(message)

	@property
	def df(self) -> pd.DataFrame:
		"""Get the Dataframe of clusters."""
		return self._df.copy()
	
	def update_clusters_from_vehicles(self, vehicles: Vehicles) -> None:
		"""
		Update clusters DataFrame based on current vehicle DataFrame.

		Parameters
		----------
		vehicles : Vehicles
			Vehicles instance with vehicle data including 'id_cluster' column.
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

class LocationsSchema(pa.DataFrameModel):
	"""Pandera schema for Logbooks Dataframe validation."""
	location: int = pa.Field(ge=0, coerce=True)
	label: str = pa.Field(coerce=True)

class Locations:
	"Class representing locations used in journeys."
	def __init__(self, input_df: pd.DataFrame = None):
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
		if input_df is None:
			self._df = LocationsSchema.example(size=0)
		else:
			self.df = input_df

	@property
	def df(self) -> pd.DataFrame:
		"""Get the Dataframe of locations."""
		return self._df.copy()
	
	@df.setter
	def df(self, value: pd.DataFrame):
		"""Set locations DataFrame with validation."""
		self._df = LocationsSchema.validate(value)