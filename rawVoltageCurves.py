import os
import seaborn as sns
import matplotlib.pyplot as plt
from correlation import process_battery_data  # This file should define and expose the battery_dfs dictionary

import os
import seaborn as sns
import matplotlib.pyplot as plt

# Access the dictionary of DataFrames
battery_dfs = process_battery_data.battery_dfs

# Define the folder where plots will be saved
output_folder = os.path.join('raw_stats', 'graphs', 'raw_voltage_curves')
os.makedirs(output_folder, exist_ok=True)

# Configure seaborn for an academic style: deep color palette, white background with grey grid lines
sns.set_theme(style="whitegrid", palette="deep")

# Iterate over each battery type DataFrame
for battery_type, df in battery_dfs.items():
    # Use the 'currents' column for grouping. If not present, skip this battery type.
    if 'currents' not in df.columns:
        print(f"Battery type '{battery_type}' does not have a 'currents' column. Skipping.")
        continue

    # For grouping by current, get the unique current values.
    # Assumes current values are strings like "30mA", "50mA", "100mA".
    unique_currents = sorted(df['currents'].unique(), key=lambda x: int(x.rstrip('mA')))
    
    # For each current condition, produce one graph
    for current_val in unique_currents:
        # Filter the DataFrame for the current condition
        df_current = df[df['currents'] == current_val]
        
        # Check for the required columns: 'soc', 'final_voltage', and 'temperature'
        required_cols = ['soc', 'final_voltage', 'temperature']
        if not all(col in df_current.columns for col in required_cols):
            print(f"Missing required columns in '{battery_type}' for current {current_val}. Skipping.")
            continue
        
        # Get unique temperature values, sorted numerically (assumes format like "25C")
        unique_temps = sorted(df_current['temperature'].unique(), key=lambda x: int(x.rstrip('C')))
        
        # Create a new figure with academic dimensions
        plt.figure(figsize=(10, 6))
        
        # For each temperature, plot final_voltage (Y-axis) versus soc (X-axis)
        for temp_val in unique_temps:
            # Filter the data for the given temperature
            df_temp = df_current[df_current['temperature'] == temp_val]
            # Sort the data by soc to create a continuous line
            df_temp = df_temp.sort_values('soc')
            plt.plot(df_temp['soc'], df_temp['final_voltage'],
                     marker='o', label=f"{temp_val}")
        
        # Add title and axis labels
        plt.title(f"{battery_type} Battery - {current_val}: Final Voltage vs. SoC", fontsize=14)
        plt.xlabel("State of Charge (SoC)", fontsize=12)
        plt.ylabel("Final Voltage", fontsize=12)
        plt.legend(title="Temperature")
        # Add grey dashed grid lines for an academic look
        plt.grid(True, color='grey', linestyle='--', linewidth=0.5)
        # Adjust layout tightly to minimize white space
        plt.tight_layout()
        
        # Define the filename for the plot and save it in PNG format in the output folder
        filename = f"{battery_type}_{current_val}_voltage_soc.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, format="png", bbox_inches="tight")
        plt.close()
        
        print(f"Saved graph for battery '{battery_type}', current '{current_val}' as: {filepath}")
