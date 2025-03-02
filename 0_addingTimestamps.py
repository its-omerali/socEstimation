#This file must be run only once if timestamps are needed to be fixed. 
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

# Define the folder containing CSV files
folder_path = 'data/'  # Adjust if necessary
csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

# Create folder for saving statistics if it doesn't exist
stats_folder = 'raw_stats'
os.makedirs(stats_folder, exist_ok=True)

# Define the base timestamp for the first file: 1 June 2020 00:00:00
# That is when actual experiments started. Since each experiment took almost 2-3 days, and they
# were running in parallel, I took a 2 days delay in my recordsheet.
base_time_first = datetime(2020, 6, 1, 0, 0, 0)

# ------------------------------------------------------------------------------
# Part 1: Add timestamp columns to each CSV file
# ------------------------------------------------------------------------------

# Iterate over each CSV file; each file's base time increases by 2 days.
for idx, file_path in enumerate(csv_files):
    # Compute file-specific base timestamp (2 days delay per file)
    file_base_time = base_time_first + timedelta(days=2 * idx)
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Determine the number of rows in the file
    n_rows = df.shape[0]
    
    # Create human-readable timestamps (ISO 8601) with a 30-second interval between rows
    timestamp_str_list = [(file_base_time + timedelta(seconds=30 * i)).isoformat() for i in range(n_rows)]
    
    # Create numerical timestamps: seconds elapsed since the file_base_time
    # All sensor readings were taken at 30 seconds interval for each experiment run. 
    # At the time of recording, timestamps were not exclusively marked, therefore, the following code
    # is used to now add timestamps with 30 seconds interval, for each file (experiment run).
    timestamp_num_list = [30 * i for i in range(n_rows)]
    
    # Add the new timestamp columns to the DataFrame
    df['timestamp_str'] = timestamp_str_list # Standard format for human readability
    df['timestamp_num'] = timestamp_num_list # Numeric format for ML model training
    
    # Save the updated DataFrame back to the CSV file (overwriting the original)
    df.to_csv(file_path, index=False)
    print(f"Updated file: {file_path} with {n_rows} rows and added timestamp_str and timestamp_num.")
    
# Part 1a. Adding Current and Temperature columns for each CSV file. Although this may seem irrelevant
# and is not going to be used for training the models. However for initial analysis, it will be
# helpful to group items. Since the next steps will involve merging all the files (per battery type) into
# one dataframe, these columns will then help to group based on either current or temperature.


# Set the folder containing the CSV files
folder_path = 'data/'  # Adjust the path as needed
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Loop over each CSV file in the folder
for file_path in csv_files:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the filename and split it by underscore to get the parts
    base_filename = os.path.basename(file_path)
    parts = base_filename.split('_')
    
    # Ensure that there are at least three parts in the filename.
    if len(parts) >= 3:
        # parts[1] corresponds to current, parts[2] corresponds to temperature.
        current_value = parts[1]        # e.g., "50mA"
        temperature_value = parts[2]      # e.g., "25C"
    else:
        # If filename format is not as expected, use defaults or flag as Unknown.
        current_value = 'Unknown'
        temperature_value = 'Unknown'
    
    # Add new columns to the DataFrame.
    # Note: The columns are added as strings; you can convert them if needed.
    df['currents'] = current_value
    df['temperature'] = temperature_value
    
    # Save the updated DataFrame back to the CSV file (overwriting the original)
    df.to_csv(file_path, index=False)
    print(f"Updated file '{base_filename}' with 'currents' = {current_value} and 'temperature' = {temperature_value}.")
