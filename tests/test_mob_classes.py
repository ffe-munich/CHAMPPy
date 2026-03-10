"""Test of mob classesand their methods."""

import pandas as pd
from champpy.core.mobility.mobility_data import MobProfilesExtended


def test_mobdata_creation(mob_profiles1):
    assert not mob_profiles1.logbooks.df.empty
    assert not mob_profiles1.vehicles.df.empty


def test_mobdata_logbook_only(mob_profiles2):
    assert not mob_profiles2.logbooks.df.empty
    assert mob_profiles2.vehicles.df is not None


def test_delete_and_add_journeys(mob_profiles1):
    id_to_delete = [2, 3]
    logbook_df_backup = mob_profiles1.logbooks.df[
        mob_profiles1.logbooks.df["id_journey"].isin(id_to_delete)
    ].copy()
    # Delete journeys
    mob_profiles1.logbooks.delete_journeys(id_journey=id_to_delete)
    assert not mob_profiles1.logbooks.df["id_journey"].isin(id_to_delete).any()
    # Add journeys back
    mob_profiles1.logbooks.add_journeys(input_df=logbook_df_backup)
    for _, row in logbook_df_backup.iterrows():
        assert (
            (mob_profiles1.logbooks.df["id_vehicle"] == row["id_vehicle"])
            & (mob_profiles1.logbooks.df["dep_dt"] == row["dep_dt"])
            & (mob_profiles1.logbooks.df["arr_dt"] == row["arr_dt"])
        ).any()


def test_reindexing(mob_profiles2):
    mob_profiles = mob_profiles2
    mob_profiles.reindexing("id_journey")
    assert mob_profiles.logbooks.df["id_journey"].is_monotonic_increasing
    assert mob_profiles.logbooks.df["id_journey"].iloc[0] == 1
    mob_profiles.reindexing("id_vehicle")
    assert mob_profiles.vehicles.df["id_vehicle"].is_monotonic_increasing
    assert mob_profiles.vehicles.df["id_vehicle"].iloc[0] == 1
    mob_profiles.reindexing("id_cluster")
    assert mob_profiles.clusters.df["id_cluster"].is_monotonic_increasing
    assert mob_profiles.clusters.df["id_cluster"].iloc[0] == 1

    # change id_cluster values to test 'all' reindexing
    mob_profiles_to_update = mob_profiles.vehicles.df.iloc[50:, :]
    mob_profiles_to_update.loc[:, "id_cluster"] = 5
    mob_profiles.vehicles.update_vehicles(input_df=mob_profiles_to_update)
    mob_profiles.reindexing("all")
    assert mob_profiles.clusters.df["id_cluster"].is_monotonic_increasing
    assert mob_profiles.clusters.df["id_cluster"].iloc[0] == 1
    assert (
        mob_profiles.vehicles.df["id_cluster"].max()
        == mob_profiles.clusters.df["id_cluster"].max()
    )
    # Add further checks as needed


def test_update_journeys(mob_profiles2):
    mob_profiles = mob_profiles2
    update_df = mob_profiles.logbooks.df[
        mob_profiles.logbooks.df["id_journey"].isin([1, 4])
    ].copy()
    update_df.loc[:, "distance"] = 20.0
    mob_profiles.logbooks.update_journeys(input_df=update_df)
    assert (
        mob_profiles.logbooks.df.set_index("id_journey").loc[[1, 4], "distance"] == 20.0
    ).all()


def test_temp_res(mob_profiles2):
    mob_profiles2.logbooks.temp_res = 0.5
    assert mob_profiles2.logbooks.temp_res == 0.5


def test_update_cluster_labels(mob_profiles3):
    clusters_df = mob_profiles3.clusters.df
    clusters_df.loc[clusters_df["id_cluster"] == 1, "label"] = "Cluster_A"
    mob_profiles3.clusters.update_clusters(input_df=clusters_df)
    assert (
        mob_profiles3.clusters.df.loc[
            mob_profiles3.clusters.df["id_cluster"] == 1, "label"
        ]
        == "Cluster_A"
    ).all()


def test_update_location_labels(mob_profiles1):
    locations_df = mob_profiles1.locations.df
    locations_df.loc[locations_df["location"] == 3, "label"] = "Further location"
    mob_profiles1.locations.update_locations(input_df=locations_df)


def test_mobdata_extended(mob_profiles2):
    mob_profiles_extended = MobProfilesExtended(mob_profiles2)
    assert hasattr(mob_profiles_extended, "df")
    assert not mob_profiles_extended.df.empty
    # difference between end_dt start_dt of each row should be <= 1 day
    assert (
        mob_profiles_extended.df["end_dt"] - mob_profiles_extended.df["start_dt"]
        <= pd.Timedelta(days=1)
    ).all()


def test_mobdata_extended_withoutsplitting(mob_profiles2):
    mob_profiles_extended = MobProfilesExtended(mob_profiles2, splitdays=False)
    assert hasattr(mob_profiles_extended, "df")
    assert not mob_profiles_extended.df.empty
