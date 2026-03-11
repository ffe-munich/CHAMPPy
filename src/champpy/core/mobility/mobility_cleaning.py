import pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple
from champpy.core.mobility.mobility_data import MobProfiles
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LimitConfig:
    """Configuration for a single limit parameter in data cleaning."""

    min_value: float = 0
    """Minimum value threshold (default: ``0``)."""

    min_method: Literal["delete"] = "delete"
    """Method to handle values below minimum (default: ``"delete"``)."""

    max_value: float = float("inf")
    """Maximum value threshold (default: ``inf``)."""

    max_method: Literal["delete", "cap"] = "cap"
    """Method to handle values above maximum: ``"delete"`` or ``"cap"`` (default: ``"cap"``)."""


@dataclass(frozen=True)
class UserParamsCleaning:
    """
    User parameters for cleaning MobProfiles.

    Configuration for data quality checks including speed, duration, distance limits
    and temporal resolution settings.
    """

    speed: LimitConfig = LimitConfig(min_value=0.01, min_method="delete", max_value=120.0, max_method="cap")
    """Speed limits configuration in km/h.
    
    Default: ``LimitConfig(min_value=0.01, max_value=120.0)``
    """

    duration: LimitConfig = LimitConfig(min_value=0.25, min_method="delete", max_value=8.0, max_method="cap")
    """Duration limits configuration in hours.
    
    Default: ``LimitConfig(min_value=0.25, max_value=8.0)``
    """

    distance: LimitConfig = LimitConfig(min_value=0.5, min_method="delete", max_value=500.0, max_method="cap")
    """
    Distance limits configuration in kilometers. 
    
    Default: ``LimitConfig(min_value=0.5, max_value=500.0)``
    """

    temp_res: float = 0.25
    """
    Temporal resolution in hours for resampling during cleaning.
    
    Default: ``0.25`` (15-minute resolution)
    """

    print_summary: bool = True
    """
    Whether to print cleaning summary to logger.
    
    Default: ``True``
    """


class MobProfilesCleaner:
    """
    Cleaner for MobProfiles with configurable limits.

    This class provides configurable data cleaning for mobility profiles, including
    removal or capping of outliers in speed, duration, and distance. It also validates
    and corrects first/last journey locations.

    Parameters
    ----------
    user_params : UserParamsCleaning, optional
        Cleaning limits. If None, default limits from :class:`UserParamsCleaning` are used.

    Attributes
    ----------
    modified_id_journeys : dict
        Dictionary tracking modified journeys by type of modification (distance, speed, duration, location).
    deleted_id_journeys : dict
        Dictionary tracking deleted journeys by type of modification (distance, speed, duration).
    params : UserParamsCleaning
        User parameters for cleaning.

    Examples
    --------
    Create a cleaner with default limits:

    >>> from champpy import MobProfilesCleaner, UserParamsCleaning
    >>> cleaner = MobProfilesCleaner()
    >>> cleaned_profiles = cleaner.clean(mob_profiles)

    Create a cleaner with custom limits:

    >>> from champpy import LimitConfig, UserParamsCleaning
    >>> custom_params = UserParamsCleaning(
    ...     speed=LimitConfig(min_value=0.5, max_value=100.0, max_method="cap"),
    ...     duration=LimitConfig(min_value=0.1, max_value=12.0, max_method="cap"),
    ...     distance=LimitConfig(min_value=0.1, max_value=600.0, max_method="cap"),
    ...     temp_res=0.5,
    ...     print_summary=True
    ... )
    >>> cleaner = MobProfilesCleaner(custom_params)
    >>> cleaned_profiles = cleaner.clean(mob_profiles)
    """

    def __init__(self, user_params: UserParamsCleaning = None):
        """
        Initialize MobProfilesCleaner with limits. See Class docstring for parameters.
        """

        # initialize variables
        self.params = user_params or UserParamsCleaning()
        self.modified_id_journeys = {
            "distance": [],
            "speed": [],
            "duration": [],
            "location": [],
        }
        self.deleted_id_journeys = {"distance": [], "speed": [], "duration": []}

    def clean(self, mob_profiles: MobProfiles) -> MobProfiles:
        """
        Clean the input MobProfiles based on configured limits.

        This method applies the following cleaning steps:

        - Resample to specified temporal resolution
        - Reindex IDs for consistency
        - Clean first/last journey locations
        - Apply limits to duration, speed, and distance
        - Log cleaning summary

        Parameters
        ----------
        mob_profiles : MobProfiles
            MobProfiles instance to clean.

        Returns
        -------
        MobProfiles
            Cleaned MobProfiles instance with ``_is_cleaned`` flag set to ``True``.
        """
        # do nothung if input is empty
        if mob_profiles.logbooks.df is None or mob_profiles.logbooks.df.empty:
            return mob_profiles

        # Resample to temporal resolution
        mob_profiles.logbooks.temp_res = self.params.temp_res

        # Reindex id_journey, id_vehicle, id_cluster
        mob_profiles.reindexing("all")

        # Ensure first/last journeys start/end at plausible locations
        self._clean_first_last_locations(mob_profiles)

        # Apply cleaning to duration, speed, and distance
        self._clean_column(mob_profiles, "duration")
        self._clean_column(mob_profiles, "speed")
        self._clean_column(mob_profiles, "distance")

        # Reindex id_journey, id_vehicle, id_cluster
        mob_profiles.reindexing("all")

        # Print summary of cleaning
        self._log_summary()

        # mark as cleaned
        mob_profiles._is_cleaned = True

        return mob_profiles

    def _clean_column(self, mob_profiles: MobProfiles, column: str, update_column: str = None) -> MobProfiles:
        """
        Generic method to clean a specific column based on limits.

        Removes journeys with values below the minimum threshold or applies capping/deletion
        to values above the maximum threshold based on the configured method.

        Parameters
        ----------
        mob_profiles : MobProfiles
            MobProfiles instance to clean.
        column : str
            Column name to check against limits (e.g., ``"distance"``, ``"speed"``, ``"duration"``).
        update_column : str, optional
            Column to update when capping (default: same as column).
            Used when derived columns (e.g., speed) require updates to base columns (e.g., distance).

        Returns
        -------
        MobProfiles
            Cleaned MobProfiles instance.
        """

        lb_df = mob_profiles.logbooks.df
        limit_config = getattr(self.params, column)

        # Identify rows above and below limits
        mask_below = lb_df[column] < limit_config.min_value
        mask_above = lb_df[column] > limit_config.max_value

        # Handle values below min_value
        if mask_below.any():
            id_journeys_to_delete = lb_df.loc[mask_below, "id_journey"].tolist()
            mob_profiles.logbooks.delete_journeys(id_journeys_to_delete)
            self.deleted_id_journeys[column].extend(id_journeys_to_delete)

        # Handle values above max_value
        if mask_above.any():
            if limit_config.max_method == "cap":
                update_df = lb_df.loc[mask_above].copy()

                # Calculate new value based on column type
                if column == "speed":
                    update_df["distance"] = limit_config.max_value * update_df.duration
                elif column == "duration":
                    update_df["arr_dt"] = update_df["dep_dt"] + pd.to_timedelta(limit_config.max_value, unit="h")
                else:
                    update_df[update_column] = limit_config.max_value

                mob_profiles.logbooks.update_journeys(update_df)
                self.modified_id_journeys[column].extend(lb_df.loc[mask_above, "id_journey"].tolist())

            elif limit_config.max_method == "delete":
                id_journeys_to_delete = lb_df.loc[mask_above, "id_journey"].tolist()
                mob_profiles.logbooks.delete_journeys(id_journeys_to_delete)
                self.deleted_id_journeys[column].extend(id_journeys_to_delete)

        return mob_profiles

    def _get_first_last_locations_of_day(self, mob_profiles: MobProfiles) -> Tuple[pd.DataFrame, list, list]:
        """
        Analyze first/last journey locations per weekday.

        For each weekday and location combination, count how many first trips of the day
        start at that location and how many last trips of the day end at that location.

        Parameters
        ----------
        mob_profiles : MobProfiles
            MobProfiles instance with logbooks data.

        Returns
        -------
        Tuple[pd.DataFrame, list, list]
            A tuple containing:

            - DataFrame with counts of first and last trips per weekday and location
            - List of ``id_journey`` for trips that are first trips of the day
            - List of ``id_journey`` for trips that are last trips of the day
        """
        lb_df = mob_profiles.logbooks.df
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
        merged_counts = pd.merge(
            first_counts,
            last_counts,
            left_on=["weekday", "dep_loc"],
            right_on=["weekday", "arr_loc"],
            how="outer",
        ).fillna(0)

        id_journeys_first = first_of_day["id_journey"].tolist()
        id_journeys_last = last_of_day["id_journey"].tolist()

        return merged_counts, id_journeys_first, id_journeys_last

    def _clean_first_last_locations(self, mob_profiles: MobProfiles) -> MobProfiles:
        """
        Analyze and correct first/last journey locations per weekday.

        For each weekday and location combination, count how many first trips of the day
        start at that location and how many last trips of the day end at that location.
        Replace implausible first/last locations with the most frequent location for that weekday.

        Parameters
        ----------
        mob_profiles : MobProfiles
            MobProfiles instance with logbooks data.

        Returns
        -------
        MobProfiles
            Updated MobProfiles instance with corrected locations.
        """
        # Abort if logbooks is empty
        lb_df = mob_profiles.logbooks.df
        if lb_df is None or lb_df.empty:
            return

        # Get first/last locations of day and corresponding id_journeys
        merged_counts, id_journeys_first, id_journeys_last = self._get_first_last_locations_of_day(mob_profiles)

        # Get dataframes for first and last of day journeys
        lb_df_only_last = mob_profiles.logbooks.df.loc[mob_profiles.logbooks.df["id_journey"].isin(id_journeys_last)]
        lb_df_only_first = mob_profiles.logbooks.df.loc[mob_profiles.logbooks.df["id_journey"].isin(id_journeys_first)]

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
            first_of_day_to_update = lb_df_only_first.merge(
                only_first[["weekday", "dep_loc"]],
                left_on=["weekday", "dep_loc"],
                right_on=["weekday", "dep_loc"],
                how="inner",
            )
            # drop dep_loc and replace with most frequent dep_loc for that weekday
            first_of_day_to_update = first_of_day_to_update.drop(columns=["dep_loc"]).merge(
                most_frequent_first[["weekday", "dep_loc"]],
                left_on="weekday",
                right_on="weekday",
                how="left",
            )
            mob_profiles.logbooks.update_journeys(first_of_day_to_update)
            # Restore location continuity after updates
            mob_profiles.logbooks.restore_location_continuity(target="dep")
            # Update first_loc in vehicle dataframe - only for affected vehicles
            new_first_locs = (
                first_of_day_to_update[["id_vehicle", "dep_loc"]]
                .drop_duplicates()
                .rename(columns={"dep_loc": "first_loc"})
            )
            # Get full vehicle data for affected vehicles and update first_loc
            vehicle_updates = mob_profiles.vehicles.df[
                mob_profiles.vehicles.df["id_vehicle"].isin(new_first_locs["id_vehicle"])
            ].copy()
            vehicle_updates = (
                vehicle_updates.drop(columns=["first_loc"])
                .set_index("id_vehicle")
                .join(new_first_locs.set_index("id_vehicle"))
                .reset_index()
            )
            mob_profiles.vehicles.update_vehicles(vehicle_updates)
            # log modified id_journeys
            self.modified_id_journeys["location"].extend(first_of_day_to_update.index.tolist())

        # Update arr_loc if journey is in only_last
        if not only_last.empty:
            # extrat rows from last_of_day where arr_loc match only_last
            last_of_day_to_update = lb_df_only_last.merge(
                only_last[["weekday", "arr_loc"]],
                left_on=["weekday", "arr_loc"],
                right_on=["weekday", "arr_loc"],
                how="inner",
            )
            # drop arr_loc and replace with most frequent arr_loc for that weekday
            last_of_day_to_update = last_of_day_to_update.drop(columns=["arr_loc"]).merge(
                most_frequent_last[["weekday", "arr_loc"]],
                left_on="weekday",
                right_on="weekday",
                how="left",
            )
            mob_profiles.logbooks.update_journeys(last_of_day_to_update)
            # Restore location continuity after updates
            mob_profiles.logbooks.restore_location_continuity(target="arr")
            # log modified id_journeys
            self.modified_id_journeys["location"].extend(last_of_day_to_update.index.tolist())

        return mob_profiles

    def _clean_transition_between_days(self, mob_profiles: MobProfiles) -> MobProfiles:
        """
        Clean logbooks to achieve consistent transition of first/last journey locations.

        The number of last journeys of a weekday arriving at a location must
        match the number of first journeys of the next weekday departing from that location.

        .. note::
           This method is not yet implemented.

        Parameters
        ----------
        mob_profiles : MobProfiles
            MobProfiles instance with logbooks data.

        Returns
        -------
        MobProfiles
            Updated MobProfiles instance.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """

        # get first/last locations of day and corresponding id_journeys
        merged_counts, id_journeys_first, id_journeys_last = self._get_first_last_locations_of_day(mob_profiles)

        # throw error as it is not implemented yet
        raise NotImplementedError("Method _allign_first_last_locations is not implemented yet.")

        # TODO: Implement alignment logic here

        return mob_profiles

    def _log_summary(self) -> None:
        """Print a summary of cleaning actions taken to the logger."""

        # Abort if printing is disabled
        if not self.params.print_summary:
            return

        # Log summary of cleaning
        logger.info("MobProfiles Cleaning Summary:")
        logger.info(f"Data has been converted to temporal resolution of {self.params.temp_res:.2f} hours.")

        # Log deleted journeys summary
        self._log_summary_method("delete")

        # Log modified journeys summary
        self._log_summary_method("modify")

        logger.info("Check deleted_id_journeys and modified_id_journeys attribute for full list.")

    def _log_summary_method(self, method: str) -> None:
        """
        Log a summary of cleaning actions taken for a specific method.

        Parameters
        ----------
        method : str
            Type of action: ``"delete"`` or ``"modify"``.

        Raises
        ------
        ValueError
            If method is not ``"delete"`` or ``"modify"``.
        """

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
