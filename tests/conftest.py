"""Pytest configuration and fixtures for tests for champpy."""

import pytest
import pandas as pd
import champpy.core.mobility.mobility_data as mob
from champpy.utils.logging import setup_logging
import os


def pytest_configure(config):
    """Pytest configuration hook to set up logging before tests run."""
    setup_logging()


@pytest.fixture
def raw_logbook_df():
    """Fixture to load example raw logbook DataFrame from CSV."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logbook_path = os.path.join(base_dir, "data", "example1", "t_logbook.csv")
    return pd.read_csv(logbook_path, parse_dates=["dep_dt", "arr_dt"])


@pytest.fixture
def raw_vehicle_df():
    """Fixture to load example raw vehicle DataFrame from CSV."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vehicle_path = os.path.join(base_dir, "data","example1", "t_vehicle.csv")
    return pd.read_csv(vehicle_path, parse_dates=["first_day", "last_day"])


@pytest.fixture
def mob_data1(raw_logbook_df, raw_vehicle_df):
    """MobData fixture created with both logbook and vehicle data."""
    return mob.MobData(input_logbooks_df=raw_logbook_df, input_vehicles_df=raw_vehicle_df)


@pytest.fixture
def mob_data2(raw_logbook_df):
    """MobData fixture created just with logbook data."""
    return mob.MobData(input_logbooks_df=raw_logbook_df)


@pytest.fixture
def mob_data3(raw_logbook_df, raw_vehicle_df):
    """MobData fixture with specific cluster assignments."""
    raw_vehicle_df.loc[20:30, "id_cluster"] = 2
    raw_vehicle_df.loc[40:50, "id_cluster"] = 5
    mob_data = mob.MobData(input_logbooks_df=raw_logbook_df, input_vehicles_df=raw_vehicle_df)
    return mob_data
