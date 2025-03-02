import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from correlation import process_battery_data  # This module should expose the dictionary `battery_dfs`

# Access the dictionary of DataFrames keyed by battery type (e.g., 'LiIon', 'LiPo', 'NiMH')
battery_dfs = process_battery_data.battery_dfs

# Define output folder for smoothing plots and create it if it doesn't exist
output_folder = os.path.join('raw_stats', 'graphs', 'smoothing')
os.makedirs(output_folder, exist_ok=True)

# Set up seaborn theme with an academic style (deep palette, grey dashed grid lines)
sns.set_theme(style="whitegrid", palette="deep")

# Default Savitzky-Golay filter parameters
default_window_length = 11  # Must be odd
default_polyorder = 2

# Iterate over each battery type DataFrame
for battery_type, df in battery_dfs.items():
    # Check that 'currents' column exists (if not, skip this battery type)
    if 'currents' not in df.columns:
        print(f"Battery type '{battery_type}' does not have a 'currents' column. Skipping.")
        continue

    # Get unique current conditions, sorting numerically (assumes format like "30mA")
    unique_currents = sorted(df['currents'].unique(), key=lambda x: int(x.rstrip('mA')))
    
    # For each current condition, produce one graph with side-by-side subplots
    for current_val in unique_currents:
        # Filter the DataFrame for the current condition
        df_current = df[df['currents'] == current_val]
        
        # Verify that required columns are present
        required_cols = ['soc', 'final_voltage', 'temperature']
        if not all(col in df_current.columns for col in required_cols):
            print(f"Missing required columns in '{battery_type}' for current {current_val}. Skipping.")
            continue
        
        # Get unique temperatures (e.g., "0C", "15C", "25C", "35C", "45C"), sorted numerically
        unique_temps = sorted(df_current['temperature'].unique(), key=lambda x: int(x.rstrip('C')))
        
        # Create a figure with two subplots side-by-side (original vs. smoothed)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot Original Data on Left Subplot
        for temp_val in unique_temps:
            df_temp = df_current[df_current['temperature'] == temp_val].sort_values('soc')
            axs[0].plot(df_temp['soc'], df_temp['final_voltage'], marker='o', label=f"{temp_val}")
        axs[0].set_title(f"{battery_type} - {current_val} (Original)", fontsize=14)
        axs[0].set_xlabel("State of Charge (SoC)", fontsize=12)
        axs[0].set_ylabel("Final Voltage", fontsize=12)
        axs[0].legend(title="Temperature")
        axs[0].grid(True, color='grey', linestyle='--', linewidth=0.5)
        
        # Plot Smoothed Data on Right Subplot using Savitzky-Golay Filter
        for temp_val in unique_temps:
            df_temp = df_current[df_current['temperature'] == temp_val].sort_values('soc')
            n_samples = len(df_temp)
            # Adjust window_length if the sample size is too small (ensuring odd number and at least 3)
            window_length = default_window_length
            if n_samples < window_length:
                window_length = n_samples if n_samples % 2 == 1 else n_samples - 1
                if window_length < 3:
                    window_length = 3
            try:
                smoothed = savgol_filter(df_temp['final_voltage'].values, window_length=window_length, polyorder=default_polyorder)
            except Exception as e:
                print(f"Error smoothing data for {battery_type}, current {current_val}, temperature {temp_val}: {e}")
                smoothed = df_temp['final_voltage'].values
            axs[1].plot(df_temp['soc'], smoothed, marker='o', label=f"{temp_val}")
        axs[1].set_title(f"{battery_type} - {current_val} (Smoothed)", fontsize=14)
        axs[1].set_xlabel("State of Charge (SoC)", fontsize=12)
        axs[1].set_ylabel("Final Voltage", fontsize=12)
        axs[1].legend(title="Temperature")
        axs[1].grid(True, color='grey', linestyle='--', linewidth=0.5)
        
        # Set overall title and adjust layout tightly
        plt.suptitle(f"{battery_type} Battery - {current_val}: Final Voltage vs. SoC\n(Original vs. Smoothed)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure as PNG with a tight bounding box
        filename = f"{battery_type}_{current_val}_voltage_smoothing.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, format='png', bbox_inches='tight')
        plt.close()
        
        print(f"Saved smoothing plot for battery '{battery_type}', current '{current_val}' as: {filepath}")
