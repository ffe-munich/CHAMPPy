"""Full pipeline test for mobility data processing, parameterization, modeling, and validation."""

import pytest
import copy
import pandas as pd
from champpy.core.mobility.mobility_cleaning import MobDataCleaner
from champpy.core.mobility.parameterization import Parameterizer, UserParamsParameterizer
from champpy.core.mobility.mobility_model import MobModel, UserParamsMobModel
from champpy.core.mobility.mobility_validation import MobPlotter, UserParamsMobPlotter


@pytest.mark.usefixtures("raw_logbook_df", "raw_vehicle_df")
def test_full_pipeline(mob_data1):
    mob_profile = mob_data1
    assert not mob_profile.logbooks.df.empty
    assert not mob_profile.vehicles.df.empty

    # Clean mob data
    data_cleaner = MobDataCleaner()
    mob_profile_cleaned = data_cleaner.clean(mob_profile)
    assert mob_profile_cleaned is not None

    # Parameterization
    user_params = UserParamsParameterizer(
        description="Example parameterization 1",
        vehicle_type="Van",
    )
    example_parameterizer = Parameterizer(user_params)
    model_params = example_parameterizer.calc_params(mob_profile_cleaned)
    assert model_params is not None

    # Mobility model
    mob_model = MobModel(model_params=model_params)
    user_params_mob = UserParamsMobModel(
        number_vehicles=50,
        start_date=pd.Timestamp("2025-01-01-01:00:00"),
        end_date=pd.Timestamp("2025-12-31-23:00:00"),
    )
    mob_profiles = mob_model.generate_mob_profiles(user_params=user_params_mob)
    assert mob_profiles is not None

    # Validation of generated mobility profiles
    mob_data_merged = copy.copy(mob_profile_cleaned)
    mob_data_merged.add_mob_data(mob_profiles)
    clusters_df = mob_data_merged.clusters.df
    clusters_df.loc[clusters_df["id_cluster"] == 1, "label"] = "Ref"
    clusters_df.loc[clusters_df["id_cluster"] == 2, "label"] = "Model"
    mob_data_merged.clusters.update_clusters(clusters_df)
    user_params_plot = UserParamsMobPlotter(filename="mobility_profiles_plot.html")
    mobplot = MobPlotter(user_params_plot)
    mobplot.plot_mob_data(mob_data_merged)
