"""Unit tests for champpy utilities related to time handling."""

import pandas as pd
import datetime
import numpy as np
from champpy.utils.time_utils import TypeDays, get_day_index


def test_get_day_index_single_datetime():
    dt_single = datetime.datetime(2026, 1, 21, 10, 30, 0)
    index_single = get_day_index(dt=dt_single, temp_res=0.25)
    assert isinstance(index_single, pd.Series)
    assert index_single[0] == 42


def test_get_day_index_single_timestamp():
    dt_single = pd.Timestamp("2026-01-21 10:30:00")
    index_single = get_day_index(dt=dt_single, temp_res=0.25)
    assert isinstance(index_single, pd.Series)
    assert index_single[0] == 42


def test_get_day_index_series():
    dt_series = pd.Series([pd.Timestamp("2026-01-21 10:00:00"), pd.Timestamp("2026-01-21 15:30:00")])
    index_series = get_day_index(dt=dt_series, temp_res=0.25)
    assert isinstance(index_series, pd.Series)
    assert len(index_series) == 2
    # Check plausible values
    assert index_series[0] < index_series[1]


def test_typedays_weekday2typeday_series():
    typedays = TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]])  # 0=weekday, 1=weekend
    weekday = pd.Series([0, 1, 2, 3, 4, 5, 6, 0, 6])
    index_typedays = typedays.weekday2typeday(weekday)
    assert isinstance(index_typedays, np.ndarray)
    assert set(index_typedays) <= {0, 1}


def test_typedays_weekday2typeday_single():
    typedays = TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]])
    weekday_single = 3
    index_typeday_single = typedays.weekday2typeday(weekday_single)
    assert index_typeday_single in [0, 1]


def test_typedays_weekday2typeday_np_array():
    typedays = TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]])
    weekday_np_array = np.array([0, 1, 2, 3, 4, 5, 6])
    index_typedays_np = typedays.weekday2typeday(weekday_np_array)
    assert isinstance(index_typedays_np, np.ndarray)
    assert set(index_typedays_np) <= {0, 1}
