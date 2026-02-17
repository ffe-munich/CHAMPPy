"""
Light tests for loading existing parameters and mobility profile generation using MobModel.
Ensures that parameters are loaded correctly and mobility profiles are generated as expected.
"""

import os
from champpy.core.mobility.parameterization import ParamsLoader
from champpy.core.mobility.mobility_model import MobModel, UserParamsMobModel
import pandas as pd
import pytest


@pytest.fixture
def params_folder():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "params")


@pytest.fixture
def params_loader(params_folder):
    return ParamsLoader(params_folder)


def test_params_info(params_loader):
    info_df = params_loader.load_info()
    # Check that info_df is a DataFrame and not empty
    assert isinstance(info_df, pd.DataFrame)
    assert not info_df.empty


def test_mobility_model_generation(params_loader):
    my_params1 = params_loader.load_params(id_params=1)
    assert my_params1 is not None
    mob_model = MobModel(model_params=my_params1)
    user_params = UserParamsMobModel(
        number_vehicles=20,
        start_date=pd.Timestamp("2025-01-01-01:00:00"),
        end_date=pd.Timestamp("2025-12-31-23:00:00"),
        random_seed=2,
    )
    mob_profiles = mob_model.generate_mob_profiles(user_params=user_params)
    # Check that mob_profiles has a DataFrame attribute and is not empty
    assert hasattr(mob_profiles.logbooks, "df")
    assert isinstance(mob_profiles.logbooks.df, pd.DataFrame)
    assert not mob_profiles.logbooks.df.empty