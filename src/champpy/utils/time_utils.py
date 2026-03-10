import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _parse_datetime(dt) -> pd.Series:
    """Parse input to pd.Timestamp.
    Args:
        dt: Input datetime as string, pd.Timestamp, or datetime.datetime
    Returns:
        pd.Series: Series of pd.Timestamp
    """
    try:
        if isinstance(dt, pd.DatetimeIndex):
            dt_out = pd.Series(dt)
        elif not isinstance(dt, pd.Series):
            dt_out = pd.Series([dt])
        else:
            dt_out = dt
        return pd.to_datetime(dt_out)
    except Exception as e:
        mssg = f"Failed to parse datetime from input {dt}: {e}"
        logger.error(mssg)
        raise ValueError(mssg)


def get_day_index(dt: pd.Timestamp | pd.Series | str, temp_res: float) -> pd.Series:
    """
    Get the index of a time within a day based on temporal resolution.

    Args:
        dt: datetime.time, pd.Timestamp, pd.Series of timestamps, or hours as float
        temp_res: temporal resolution in hours

    Returns:
        pd.Series: index within the day (0-based)

    Example:
        temp_res = 1.0 (hourly)
        10:00 -> index = 10

        temp_res = 0.5 (30 minutes)
        10:00 -> index = 20
        10:30 -> index = 21

        # Single timestamp
        >>> get_day_index(dt = pd.Timestamp('2026-01-21 10:30:00'), temp_res = 0.25)
        42

        # Vectorized Series
        >>> get_day_index(dt = pd.Series([pd.Timestamp('2026-01-21 10:00:00'), pd.Timestamp('2026-01-21 15:30:00')]), temp_res = 0.25)
        0    40
        1    62
        dtype: int64
    """
    dt = _parse_datetime(dt)
    hour = dt.dt.hour + dt.dt.minute / 60 + dt.dt.second / 3600
    return pd.Series((hour / temp_res).astype(int), index=dt.index)


def get_week_index(dt: pd.Timestamp | pd.Series | str, temp_res: float) -> pd.Series:
    """
    Get the index within a week based on temporal resolution.

    Args:
        dt: pd.Timestamp or pd.Series of timestamps
        temp_res: temporal resolution in hours

    Returns:
        pd.Series: index within the week (0-based)

    Example:
        temp_res = 1.0 (hourly)
        Monday 10:00 -> index = 10 (0*24 + 10)
        Tuesday 15:00 -> index = 39 (1*24 + 15)

        temp_res = 0.25 (15 minutes)
        Monday 10:00 -> index = 40 (0*96 + 40)
        Tuesday 15:00 -> index = 156 (1*96 + 60)

        # Single timestamp
        >>> get_week_index(dt = pd.Timestamp('2026-01-21 10:30:00'), temp_res = 0.25)
        138  # Tuesday 10:30 (1*96 + 42)

        # Vectorized Series
        >>> get_week_index(dt = pd.Series([pd.Timestamp('2026-01-20 10:00:00'), pd.Timestamp('2026-01-21 15:30:00')]), temp_res = 0.25)
        1    158  # Tuesday 15:30 (1*96 + 62)
        dtype: int64
    """
    # Parse datetime
    dt = _parse_datetime(dt)

    # Determine weekday index
    weekday = dt.dt.dayofweek  # Monday=0, Sunday=6
    day_idx = get_day_index(dt, temp_res)
    indices_per_day = int(24 / temp_res)
    return pd.Series(weekday * indices_per_day + day_idx, index=dt.index)


def get_datetime_array(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    temp_res: float,
    number_days_buffer: int = 0,
) -> tuple[pd.DatetimeIndex, pd.Series]:
    """
    Create a datetime array with buffer days before and after the actual period.
    Returns (dt_array, mask_buffer).

    Parameters:
        start_date (pd.Timestamp): Start date of the actual period.
        end_date (pd.Timestamp): End date of the actual period.
        temp_res (float): Temporal resolution in hours.
        number_days_buffer (int): Number of buffer days before and after the period.
    """
    start_dt_wo_buffer = start_date.normalize()
    start_dt = start_dt_wo_buffer - pd.Timedelta(days=number_days_buffer)
    end_dt_wo_buffer = (
        end_date.normalize() + pd.Timedelta(days=1) - pd.Timedelta(hours=temp_res)
    )
    end_dt = end_dt_wo_buffer + pd.Timedelta(days=number_days_buffer)
    frequency = f"{temp_res}h"
    dt_array = pd.date_range(start=start_dt, end=end_dt, freq=frequency)

    mask_buffer = (dt_array < start_dt_wo_buffer) | (dt_array > end_dt_wo_buffer)
    return dt_array, mask_buffer


class TypeDays:
    """
    Utility class to group weekdays into typedays.

    The class allows to define custom groups of weekdays (e.g., Mon-Fri, Sat-Sun) and provides methods to convert weekday indices to typeday indices.

    Parameters
    ----------
    groups : list[list[int]], optional
        List of weekday groups. Each inner list defines one typeday and contains
        weekday indices (``0=Monday`` ... ``6=Sunday``).

    Attributes
    ----------
    groups : list[list[int]]
        The defined groups of weekdays for each typeday.
    index : list[int]
        The index of each typeday (0-based).
    names : list[str]
        The names of each typeday based on the grouped weekdays (e.g., "Mon-Fri", "Sat-Sun").
    number : int
        The number of typedays defined.

    Examples
    --------
    >>> typedays = TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]])
    >>> typedays.index
    [0, 1]
    >>> typedays.names
    ['Mon-Fri', 'Sat-Sun']

    >>> TypeDays(groups=[[0, 1, 2, 3, 4], [5], [6]]).names
    ['Mon-Fri', 'Sat', 'Sun']

    >>> TypeDays(groups=[[0, 2, 4], [1, 3, 5, 6]]).names
    ['Mon-Fri', 'Tue-Sun']
    """

    def __init__(self, groups: list[list[int]] = [[0], [1], [2], [3], [4], [5], [6]]):
        """
        See Class docstring for parameters and example.
        """
        # Validate groups
        self._validate_groups(groups)

        self.groups = groups

        # Save index of typedays for quick lookup
        self.index = list(range(0, len(self.groups)))

        # save number of TypeDays
        self.number = len(self.groups)

        # save names of typedays
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        # generate names based on groups: Mon-Fri, Sat-Sun
        self.names = []
        for group in groups:
            if len(group) == 1:
                name = weekday_names[group[0]]
            else:
                name = f"{weekday_names[group[0]]}-{weekday_names[group[-1]]}"
            self.names.append(name)

    def _validate_groups(self, groups: list[list[int]]):
        """Validate typeday groups."""
        # Check validity of groups
        if groups is None or not isinstance(groups, list) or len(groups) == 0:
            mssg = "Typeday groups must be a non-empty list of lists."
            logger.error(mssg)
            raise ValueError(mssg)
        for group in groups:
            if not isinstance(group, list) or len(group) == 0:
                mssg = "Each typeday group must be a non-empty list of integers representing weekdays."
                logger.error(mssg)
                raise ValueError(mssg)
            for day in group:
                if day < 0 or day > 6:
                    mssg = f"Invalid weekday {day} in groups. Must be between 0 (Monday) and 6 (Sunday)."
                    logger.error(mssg)
                    raise ValueError(mssg)
        # validate that all days are covered and no duplicates
        all_days = [day for group in groups for day in group]
        if sorted(all_days) != list(range(7)):
            mssg = "All days (0-6) must be covered exactly once in typeday groups."
            logger.error(mssg)
            raise ValueError(mssg)

    def weekday2typeday(
        self, index_weekday: int | pd.Series | np.ndarray
    ) -> int | np.ndarray:
        """
        Convert weekday (0=Monday,..6=Sunday) to typeday index based on groups.

        Parameters
        ----------
        index_weekday : int | pandas.Series | numpy.ndarray
            Weekday index or array/series of weekday indices.

        Examples
        --------
        >>> typedays = TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]])
        >>> typedays.weekday2typeday(np.array([0, 1, 2, 3, 4, 5, 6, 4, 6]))
        [0, 0, 0, 0, 0, 1, 1, 0, 1]
        """
        if isinstance(index_weekday, int):
            # Single value
            for idx, group in enumerate(self.groups):
                if index_weekday in group:
                    return idx

        # Check for type of class
        if isinstance(index_weekday, pd.Series) or isinstance(index_weekday, pd.Index):
            # convert to numpy array for faster processing (ensure writable)
            index_weekday_array = index_weekday.to_numpy(copy=True)
        elif isinstance(index_weekday, np.ndarray):
            index_weekday_array = np.array(index_weekday, copy=True)
        else:
            raise TypeError("Input must be int, pd.Series, pd.Index, or np.ndarray.")

        for i, group in enumerate(self.groups):
            mask = np.isin(index_weekday_array, group)
            index_weekday_array[mask] = i  # 1-based
        return index_weekday_array
