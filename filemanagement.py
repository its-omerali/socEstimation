import os
import glob
import pandas as pd
from datetime import datetime, timedelta


# ------------------------------------------------------------------------------
# Part 2: Group files into separate DataFrames based on battery type in filename
# ------------------------------------------------------------------------------
folder_path = 'data/'  # Adjust if necessary. But currently all files are in this 'data' folder.
csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
# Create a dictionary to hold lists of DataFrames keyed by battery type
battery_dfs = {}

for file_path in csv_files:
    # Extract the battery type from the filename.
    # Assumption: Filename format is like 'LiIon_50mA_25C_exp1.csv'
    base_filename = os.path.basename(file_path)
    battery_type = base_filename.split('_')[0]
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list for this battery type
    battery_dfs.setdefault(battery_type, []).append(df)

# Concatenate lists of DataFrames for each battery type into a single DataFrame
for battery_type in battery_dfs:
    battery_dfs[battery_type] = pd.concat(battery_dfs[battery_type], ignore_index=True)
    print(f"Combined DataFrame for battery type '{battery_type}' now has {battery_dfs[battery_type].shape[0]} records.")

# ------------------------------------------------------------------------------
# Part 3: Drop the 'half_voltage' column in each DataFrame (if it exists)
# ------------------------------------------------------------------------------

for battery_type, df in battery_dfs.items():
    if 'half_voltage' in df.columns:
        df.drop(columns=['half_voltage'], inplace=True)
        print(f"Dropped column 'half_voltage' from {battery_type} DataFrame.")

# ------------------------------------------------------------------------------
# Part 4: Calculate statistics for each DataFrame
# ------------------------------------------------------------------------------
# Create folder for saving statistics if it doesn't exist
stats_folder = 'raw_stats'
os.makedirs(stats_folder, exist_ok=True)
for battery_type, df in battery_dfs.items():
    # Calculate descriptive statistics for numeric columns using describe()
    stats = df.describe(include='all')
    
    # Also calculate total number of records
    total_records = df.shape[0]
    stats.loc['total_records'] = total_records  # Append total record count as an extra row
    
    # Display statistics in the terminal
    print(f"\nStatistics for battery type '{battery_type}':")
    print(stats)
    
    # ------------------------------------------------------------------------------
    # Part 5: Save the statistics to a CSV file in the 'raw_stats' folder
    # ------------------------------------------------------------------------------
    
    stats_filename = os.path.join(stats_folder, f"{battery_type}_stats.csv")
    stats.to_csv(stats_filename)
    print(f"Saved statistics for '{battery_type}' to {stats_filename}.")
