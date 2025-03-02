#####################################################
# This will normalize all important columns and store pickle files for each battery type
# Later on, these normalized pickle files can be directly opened into separate dataframes
# to apply LSTM models for each type. Sometimes, model may not respond well to 
# normalized data, so in that case, the previous battery dataframe generated in 'correlation.py'
# can be used. Since no further processing was required after this stage, therefore pickle files
# were generated to speed up the process and maintain the consistency. Another advantage is that
# the actual data files will not be used and hence won't be altered accidently.
#####################################################
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from correlation import process_battery_data  # This module should expose a dictionary called battery_dfs

# Access the in-memory dictionary of dataframes keyed by battery type (e.g., 'LiIon', 'LiPo', 'NiMH')
battery_dfs = process_battery_data.battery_dfs

# Define output folder for normalized files and create it if it doesn't exist
output_folder = os.path.join('data', 'normalized')
os.makedirs(output_folder, exist_ok=True)

# List of base columns to normalize (except current which may be under "current" or "currents")
base_columns = ['timestamp_num', 'final_voltage', 'capacity', 'soc']

# Process each battery type separately
for battery_type, df in battery_dfs.items():
    print(f"Processing battery type: {battery_type}")
    
    # Determine the column for current: use 'current' if available; otherwise, 'currents'
    if 'current' in df.columns:
        current_col = 'current'
    else:
        print(f"Battery type '{battery_type}' is missing a current column. Skipping.")
        continue

    # Ensure the DataFrame has a 'temperature' column
    if 'temperature' not in df.columns:
        print(f"Battery type '{battery_type}' is missing a temperature column. Skipping.")
        continue

    # The full list of columns we want to normalize (including the current column)
    columns_to_normalize = base_columns + [current_col]

    # Create a copy of the original dataframe to add normalized columns
    norm_df = df.copy()

    # For each column to normalize, apply MinMaxScaler over the entire battery type dataframe
    for col in columns_to_normalize:
        if col in norm_df.columns:
            scaler = MinMaxScaler()
            # Reshape column to 2D array for scaler, then flatten the result back to 1D
            norm_values = scaler.fit_transform(norm_df[[col]])
            norm_df[f"{col}_norm"] = norm_values
        else:
            print(f"Column '{col}' not found in battery type '{battery_type}'.")

    # Save the entire normalized dataframe as a pickle file for later reuse.
    pickle_filename = os.path.join(output_folder, f"{battery_type}_normalized.pkl")
    with open(pickle_filename, "wb") as f:
        pickle.dump(norm_df, f)
    print(f"Saved normalized pickle for battery '{battery_type}' as: {pickle_filename}")


