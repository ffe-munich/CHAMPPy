"""Test of the validation module of champpy."""

import pandas as pd
from champpy.core.mobility.mobility_validation import MobPlotter, UserParamsMobPlotter, MobilityCharacteristics
from champpy.utils.time_utils import TypeDays


def test_mobility_characteristics_default(mob_data1):
    mob_data = mob_data1
    mob_char = MobilityCharacteristics(mob_data)
    # Check that char_df is a DataFrame and has expected columns
    assert isinstance(mob_char.df, pd.DataFrame)
    assert mob_char.df.shape[0] == 1  # Default only one row
    expected_columns = [
        "typeday",
        "daily_kilometrage",
        "daily_journey_time",
        "number_journeys_per_day",
        "share_days_with_journeys",
        "locations",
        "share_of_time_at_locations",
    ]
    assert all(col in mob_char.df.columns for col in expected_columns)


def test_mobility_characteristics_grouping_vehicle(mob_data1):
    mob_data = mob_data1
    typedays = TypeDays(groups=[[0, 1, 2, 3, 4], [5, 6]])  # weekday vs weekend
    mob_char = MobilityCharacteristics(mob_data, grouping="vehicle", typedays=typedays)
    # Check that char_df is a DataFrame and has expected number of rows
    assert mob_char.df.shape[0] == 2  # Two typdays: weekday and weekend
    assert isinstance(mob_char.df.daily_kilometrage[0], list)  # must contain a list due to vehicle grouping


def test_mobility_characteristics_grouping_day(mob_data1):
    mob_data = mob_data1
    typedays = TypeDays(groups=[[0], [1], [2], [3], [4], [5], [6]])  # every day separate except weekend combined
    mob_char = MobilityCharacteristics(mob_data, grouping="day", typedays=typedays)
    # Check that char_df is a DataFrame and has expected number of rows
    assert mob_char.df.shape[0] == 7
    assert isinstance(mob_char.df.daily_kilometrage[0], list)  # must contain a list due to day grouping


def test_mob_plotter_onecluster(mob_data1):
    mob_data = mob_data1
    user_params = UserParamsMobPlotter(show=True, save_plot=True, filename="test.dt")  # just to test initialization
    mob_plotter = MobPlotter(user_params=user_params)
    mob_plotter.plot_mob_data(mob_data)


def test_mob_plotter_nclusters(mob_data3):
    mob_data = mob_data3
    user_params = UserParamsMobPlotter(show=True, save_plot=True, filename="test.dt")  # just to test initialization
    mob_plotter = MobPlotter(user_params=user_params)
    mob_plotter.plot_mob_data(mob_data)
