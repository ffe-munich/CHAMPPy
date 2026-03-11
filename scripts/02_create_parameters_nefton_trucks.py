import pandas as pd
import os
import champpy

# Load example raw data from CSV
# --------------------------------------------------------
# Load example raw logbook data from CSV
base_dir = os.getcwd()
folder_name = "nefton_trucks"
logbook_path = os.path.join(base_dir, "data", folder_name, "t_logbook.csv")
print(logbook_path)
if not os.path.exists(logbook_path):
    raise FileNotFoundError(f"Logbook file not found: {logbook_path}")
raw_logbook_df = pd.read_csv(logbook_path, parse_dates=["dep_dt", "arr_dt"])

# Load example raw vehicle data from CSV
vehicle_path = os.path.join(base_dir, "data", folder_name, "t_vehicle.csv")
if not os.path.exists(vehicle_path):
    raise FileNotFoundError(f"Vehicle file not found: {vehicle_path}")
raw_vehicle_df = pd.read_csv(vehicle_path, parse_dates=["first_day", "last_day"], date_format="%d-%b-%Y")

# Create raw mobility profiles
raw_profiles = champpy.MobProfiles(input_logbooks_df=raw_logbook_df, input_vehicles_df=raw_vehicle_df)

# Clean the raw mobility profiles
# --------------------------------------------------------
# Initialize the data cleaner with user parameters
user_params_cleaning = champpy.UserParamsCleaning(
    speed=champpy.LimitConfig(min_value=0.01, min_method="delete", max_value=120.0, max_method="cap"),
    duration=champpy.LimitConfig(min_value=0.25, min_method="delete", max_value=8.0, max_method="cap"),
    distance=champpy.LimitConfig(min_value=0.5, min_method="delete", max_value=700.0, max_method="cap"),
    temp_res=0.25,  # Temporal resolution in hours
    print_summary=True,
)
data_cleaner = champpy.MobProfilesCleaner(user_params=user_params_cleaning)

# Apply data cleaner on your raw mobility profiles
mob_profile_cleaned = data_cleaner.clean(raw_profiles)

# Update location labels
locations_df = mob_profile_cleaned.locations.df
locations_df["label"] = ["Driving", "Depot", "Other location", "Rest area", "Industrial area"]
mob_profile_cleaned.locations.update_locations(locations_df)

# Parameterization
# --------------------------------------------------------
# Define user params for the parameterization
user_params_param = champpy.UserParamsParameterizer(
    description="Trucks based on Nefton Dataset (https://doi.org/10.5281/zenodo.7599687)",  # Define a description for the parameter set
    vehicle_type="Truck",  # Type of vehicle the parameters apply to (e.g., "Car", "Van", "Truck")
    temp_res=0.25,  # Temporal resolution in hours
)
# Create an instance of the Parameterizer
my_parameterizer = champpy.Parameterizer(user_params_param)

# calculate the model parameters based on the cleaned mobility profiles
model_params = my_parameterizer.calc_params(mob_profile_cleaned)

# Generate synthetic mobility profiles
# --------------------------------------------------------
mob_model = champpy.MobModel(model_params=model_params)
user_params_mob = champpy.UserParamsMobModel(
    number_vehicles=50,
    start_date=pd.Timestamp("2025-01-01-00:00:00"),
    end_date=pd.Timestamp("2025-12-31-23:00:00"),
)
mob_profiles = mob_model.generate_mob_profiles(user_params=user_params_mob)

# Validation
# --------------------------------------------------------
# Create a copy of the cleaned mobility profiles and add the modeld profiles for comparison using add_mob_data()
mob_data_merged = mob_profile_cleaned.copy()
mob_data_merged.add_mob_profiles(mob_profiles, old_cluster_label="Ref", new_cluster_label="Model")

# Initialize user parameters for plotting the mobiltiy profiles
user_params_plot = champpy.UserParamsMobPlotter(filename="validation_nefton_trucks.html", clustering=True)

# Create instance of the mobility plotter
mobplot = champpy.MobPlotter(user_params_plot)

# Plot the mobility profiles for the merged data (ref + model)
mobplot.plot_mob_profiles(mob_data_merged)

# Save Parameters
params_loader = champpy.ParamsLoader()
params_loader._save_params(model_params)
