import pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple
from champpy.core.mobility_data import MobData
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LimitConfig:
	"""Configuration for a single limit parameter."""
	min_value: float = 0
	min_method: Literal["delete"] = "delete"
	max_value: float = float('inf')
	max_method: Literal["delete", "cap"] = "cap"


@dataclass(frozen=True)
class UserParamsCleaning:
	"""
	User parameters for cleaning MobData.

	Parameters
	----------
	speed : LimitConfig
		Speed limits configuration in km/h.
	duration : LimitConfig
		Duration limits configuration in hours.
	distance : LimitConfig
		Distance limits configuration in kilometers.
	temp_res : float
		Temporal resolution in hours for resampling during cleaning.
	"""
	speed: LimitConfig = LimitConfig(min_value=0.01, min_method="delete", max_value=120.0, max_method="cap")
	duration: LimitConfig = LimitConfig(min_value=0.25, min_method="delete", max_value=8.0, max_method="cap")
	distance: LimitConfig = LimitConfig(min_value=0.5, min_method="delete", max_value=500.0, max_method="cap")
	temp_res: float = 0.25  # Temporal resolution in hours
	print_summary: bool = True  # Whether to print cleaning summary


class MobDataCleaner:
	"""Cleaner for MobData with configurable limits."""

	def __init__(self, user_params: UserParamsCleaning = None):
		"""
		Initialize MobDataClean with limits and clean the input data.
		
		Parameters
		----------
		input_mob_data : MobData
			MobData instance to clean.
		user_params : UserParamsCleaning, optional
			Cleaning limits. If None, default limits are used.
		"""
		
		# initialize variables
		self.params = user_params or UserParamsCleaning()
		self.modified_id_journeys = {
			"distance": [],
			"speed": [],
			"duration": [],
			"location": []
		}
		self.deleted_id_journeys = {
			"distance": [],
			"speed": [],
			"duration": []
		}

	def clean(self, mob_data: MobData) -> MobData:
		"""
		Clean the input MobData based on configured limits.

		Parameters
		----------
		mob_data : MobData
			MobData instance to clean.

		Returns
		-------
		MobData
			Cleaned MobData instance.
		"""
	    # do nothung if input is empty
		if mob_data.logbook.df is None or mob_data.logbook.df.empty:
			return mob_data
		
		# Resample to temporal resolution
		mob_data.logbook.res_min = self.params.temp_res

		# Ensure first/last journeys start/end at plausible locations
		self._clean_first_last_locations(mob_data)
		
		# Apply cleaning to duration, speed, and distance
		self._clean_column(mob_data, "duration")
		self._clean_column(mob_data, "speed")
		self._clean_column(mob_data, "distance")

		# Reset cluster ids in vehicle data starting from 0
		self._reindex_clusters(mob_data)

		# Print summary of cleaning
		self._log_summary()

		# mark as cleaned
		mob_data._is_cleaned = True

		return mob_data

	def _clean_column(self, mob_data: MobData, column: str, update_column: str = None) -> MobData:
		""" 
		Generic method to clean a specific column based on limits.
		
		Parameters
		----------
		mob_data : MobData
			MobData instance to clean.
		column : str
			Column name to check against limits (e.g., "distance", "speed", "duration").
		update_column : str, optional
			Column to update when capping (default: same as column).
			Used when derived columns (e.g., speed) require updates to base columns (e.g., distance).

		Returns
		-------
		MobData
			Cleaned MobData instance.
		"""

		lb_df = mob_data.logbook.df
		limit_config = getattr(self.params, column)

		# Identify rows above and below limits
		mask_below = lb_df[column] < limit_config.min_value
		mask_above = lb_df[column] > limit_config.max_value


		# Handle values below min_value
		if mask_below.any():
				id_journeys_to_delete = lb_df.loc[mask_below, "id_journey"].tolist()
				mob_data.logbook.delete_journeys(id_journeys_to_delete)
				self.deleted_id_journeys[column].extend(id_journeys_to_delete)

		# Handle values above max_value
		if mask_above.any():
			if limit_config.max_method == "cap":
				update_df = lb_df.loc[mask_above].copy()
				
				# Calculate new value based on column type
				if column == "speed":
					update_df["distance"] = limit_config.max_value * update_df.duration
				elif column == "duration":
					update_df["arr_dt"] = update_df["dep_dt"] + pd.to_timedelta(limit_config.max_value, unit='h')
				else:
					update_df[update_column] = limit_config.max_value
				
				mob_data.logbook.update_journeys(update_df)
				self.modified_id_journeys[column].extend(lb_df.loc[mask_above, "id_journey"].tolist())
			
			elif limit_config.max_method == "delete":
				id_journeys_to_delete = lb_df.loc[mask_above, "id_journey"].tolist()
				mob_data.logbook.delete_journeys(id_journeys_to_delete)
				self.deleted_id_journeys[column].extend(id_journeys_to_delete)

		return mob_data
	
	def _get_first_last_locations_of_day(self, mob_data: MobData) -> Tuple[pd.DataFrame, list, list]:
		"""Analyze first/last journey locations per weekday.
		
		For each weekday and location combination, count how many first trips of the day
		start at that location and how many last trips of the day end at that location.

		Args:
			mob_data (MobData): MobData instance with logbook data.

		Returns:
        Tuple[pd.DataFrame, list, list]: A tuple containing:
			- A DataFrame with counts of first and last trips per weekday and location.
			- A list of id_journeys that are first trips of the day.
			- A list of id_journeys that are last trips of the day.
		"""
		lb_df = mob_data.logbook.df
		if lb_df is None or lb_df.empty:
			return pd.DataFrame(), [], []

		# Work on a copy sorted by vehicle and dep_dt
		df = lb_df.sort_values(["id_vehicle", "dep_dt"]).reset_index(drop=True)

		# Add day and weekday information
		df["day"] = df["dep_dt"].dt.floor("D")
		df["weekday"] = df["dep_dt"].dt.dayofweek  # Monday=0, Sunday=6
		
		# Get first and last trips per day per vehicle
		first_of_day = df.sort_values("dep_dt").groupby(["id_vehicle", "day"]).first().reset_index()
		last_of_day = df.sort_values("dep_dt").groupby(["id_vehicle", "day"]).last().reset_index()

		# Count first trips by weekday and location
		first_counts = first_of_day.groupby(["weekday", "dep_loc"]).size().reset_index(name="count_first")
		
		# Count last trips by weekday and location
		last_counts = last_of_day.groupby(["weekday", "arr_loc"]).size().reset_index(name="count_last")

		# Merge first and last counts for reporting
		merged_counts = pd.merge(first_counts, last_counts, left_on=["weekday", "dep_loc"], right_on=["weekday", "arr_loc"], how="outer").fillna(0)

		id_journeys_first = first_of_day["id_journey"].tolist()
		id_journeys_last = last_of_day["id_journey"].tolist()

		return merged_counts, id_journeys_first, id_journeys_last

	
	def _clean_first_last_locations(self, mob_data: MobData) -> MobData:
		"""Analyze and report first/last journey locations per weekday.
		
		For each weekday and location combination, count how many first trips of the day
		start at that location and how many last trips of the day end at that location.
		Replace implausible first/last locations with the most frequent location for that weekday.

		Args:
			mob_data (MobData): MobData instance with logbook data.
		"""
		# Abort if logbook is empty
		lb_df = mob_data.logbook.df
		if lb_df is None or lb_df.empty:
			return
		
		# Get first/last locations of day and corresponding id_journeys
		merged_counts, id_journeys_first, id_journeys_last = self._get_first_last_locations_of_day(mob_data)

		# Get dataframes for first and last of day journeys
		lb_df_only_last = mob_data.logbook.df.loc[mob_data.logbook.df["id_journey"].isin(id_journeys_last)]
		lb_df_only_first = mob_data.logbook.df.loc[mob_data.logbook.df["id_journey"].isin(id_journeys_first)]

		# add weekday column to lb_df_only_last and lb_df_only_first
		lb_df_only_last["weekday"] = lb_df_only_last["arr_dt"].dt.dayofweek
		lb_df_only_first["weekday"] = lb_df_only_first["dep_dt"].dt.dayofweek

		# identify first or last counts without matching counts
		only_first = merged_counts[merged_counts["count_last"] == 0]
		only_last = merged_counts[merged_counts["count_first"] == 0]
		
		# merge only first on first_counts to fint the most frequent dep_loc for the smae weekday
		most_frequent_first = merged_counts.loc[merged_counts.groupby("weekday")["count_first"].idxmax()] 
		most_frequent_last = merged_counts.loc[merged_counts.groupby("weekday")["count_last"].idxmax()]

		# Update dep_loc if journey is in only_first
		if not only_first.empty:
			# extrat rows from first_of_day where dep_loc match only_first
			first_of_day_to_update = lb_df_only_first.merge(only_first[["weekday", "dep_loc"]], left_on=["weekday", "dep_loc"], right_on=["weekday", "dep_loc"], how="inner")
			# drop dep_loc and replace with most frequent dep_loc for that weekday
			first_of_day_to_update = first_of_day_to_update.drop(columns=["dep_loc"]).merge(
				most_frequent_first[["weekday", "dep_loc"]], 
				left_on="weekday", 
				right_on="weekday", 
				how="left")
			mob_data.logbook.update_journeys(first_of_day_to_update)
			# Restore location continuity after updates
			mob_data.logbook.restore_location_continuity(target="dep")
			# Update first_loc in vehicle dataframe - only for affected vehicles
			new_first_locs = first_of_day_to_update[["id_vehicle", "dep_loc"]].drop_duplicates().rename(columns={"dep_loc": "first_loc"})
			# Get full vehicle data for affected vehicles and update first_loc
			vehicle_updates = mob_data.vehicles.df[mob_data.vehicles.df["id_vehicle"].isin(new_first_locs["id_vehicle"])].copy()
			vehicle_updates = vehicle_updates.drop(columns=["first_loc"]).set_index("id_vehicle").join(
				new_first_locs.set_index("id_vehicle")
			).reset_index()
			mob_data.vehicles.update_vehicles(vehicle_updates)
			# log modified id_journeys
			self.modified_id_journeys["location"].extend(first_of_day_to_update.index.tolist())
		
		# Update arr_loc if journey is in only_last
		if not only_last.empty:
			# extrat rows from last_of_day where arr_loc match only_last
			last_of_day_to_update = lb_df_only_last.merge(only_last[["weekday", "arr_loc"]], left_on=["weekday", "arr_loc"], right_on=["weekday", "arr_loc"], how="inner")	
			# drop arr_loc and replace with most frequent arr_loc for that weekday
			last_of_day_to_update = last_of_day_to_update.drop(columns=["arr_loc"]).merge(
				most_frequent_last[["weekday", "arr_loc"]], 
				left_on="weekday", 
				right_on="weekday", 
				how="left")
			mob_data.logbook.update_journeys(last_of_day_to_update)
			# Restore location continuity after updates
			mob_data.logbook.restore_location_continuity(target="arr") 
			# log modified id_journeys
			self.modified_id_journeys["location"].extend(last_of_day_to_update.index.tolist())

		return mob_data
	
	def _clean_transition_between_days(self, mob_data: MobData) -> MobData:
		""" Clean the logbook to achieve a consistent transition of first/last journey locations per weekday.
		The number of last journeys of a weekday arriving at a location must 
		match the number of first journeys of the next weekday departing from that location.

		Args:
			mob_data (MobData): MobData instance with logbook data.
		"""

		# get first/last locations of day and corresponding id_journeys
		merged_counts, id_journeys_first, id_journeys_last = self._get_first_last_locations_of_day(mob_data)

		# throw error as it is not implemented yet
		raise NotImplementedError("Method _allign_first_last_locations is not implemented yet.")

		# TODO: Implement alignment logic here
		
		return mob_data
	
	def _reindex_clusters(self, mob_data: MobData) -> MobData:
		"""
		Reindex cluster IDs of vehiclesto be consecutive starting from 0.
		"""	
		unique_clusters = sorted(mob_data.vehicles.df["cluster"].unique())
		cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
		mob_data.vehicles._df["cluster"] = mob_data.vehicles._df["cluster"].map(cluster_mapping)

	def _log_summary(self) -> None:
		"""Print a summary of cleaning actions taken."""

		# Abort if printing is disabled
		if not self.params.print_summary:
			return
		
		# Log summary of cleaning
		logger.info("MobData Cleaning Summary:")
		logger.info(f"Data has been converted to temporal resolution of {self.params.temp_res:.2f} hours.")
		
		# Log deleted journeys summary
		self._log_summary_method("delete")
		
		# Log modified journeys summary
		self._log_summary_method("modify")
		
		logger.info("Check deleted_id_journeys and modified_id_journeys attribute for full list.")

	def _log_summary_method(self, method: str) -> None:
		"""Log a summary of cleaning actions taken."""
		
		# Select appropriate dictionary based on method
		if method == "delete":
			id_journeys = self.deleted_id_journeys
			name_action = "Deleted"
		elif method == "modify":
			id_journeys = self.modified_id_journeys
			name_action = "Modified"
		else:
			message = "method must be either 'delete' or 'modify'"
			logger.error(message)
			raise ValueError(message)

		# Build complete summary message
		summary_lines = [f"{name_action} journeys:"]
		for key, val in id_journeys.items():
			display_ids = val[:5]
			remaining = len(val) - 5
			if remaining > 0:
				display_list = display_ids + ["..."]
			else:
				display_list = display_ids
			ids_str = "[" + ", ".join(map(str, display_list)) + "]"
			summary_lines.append(f"  - Due to {key} issues: {len(val)} journeys: id_journeys = {ids_str}")
		
		# Log complete summary at once
		message = "\n".join(summary_lines)
		logger.info(message)
	
	