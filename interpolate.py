import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from correlation import process_battery_data  # This module should expose the dictionary `battery_dfs`

# Access the dictionary of DataFrames keyed by battery type (e.g., 'LiIon', 'LiPo', 'NiMH')
battery_dfs = process_battery_data.battery_dfs

# Define output folder for interpolation plots and create it if it doesn't exist
output_folder = os.path.join('raw_stats', 'graphs', 'interpolation')
os.makedirs(output_folder, exist_ok=True)

# Set up seaborn theme with academic style: whitegrid, deep palette, grey dashed grid lines
sns.set_theme(style="whitegrid", palette="deep")

# Default Savitzky-Golay filter parameters
default_window_length = 11  # Must be odd
default_polyorder = 2

# Debug: count how many graphs are created
graph_count = 0

# Iterate over each battery type DataFrame
for battery_type, df in battery_dfs.items():
    # Check for required columns
    required_main_cols = ['currents', 'soc', 'final_voltage', 'temperature']
    if not all(col in df.columns for col in required_main_cols):
        print(f"Battery type '{battery_type}' is missing one of the required columns: {required_main_cols}. Skipping.")
        continue

    # Get unique current conditions, sorted numerically (assumes format like "30mA")
    unique_currents = sorted(df['currents'].unique(), key=lambda x: int(x.rstrip('mA')))
    
    # For each current condition, produce one graph with side-by-side subplots
    for current_val in unique_currents:
        # Filter the DataFrame for the current condition
        df_current = df[df['currents'] == current_val]
        
        # Get unique temperatures (e.g., "0C", "15C", etc.), sorted numerically
        unique_temps = sorted(df_current['temperature'].unique(), key=lambda x: int(x.rstrip('C')))
        
        # Create a figure with two subplots: left for smoothed, right for interpolated
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loop over each temperature group
        for temp_val in unique_temps:
            df_temp = df_current[df_current['temperature'] == temp_val].sort_values('soc')
            if df_temp.empty:
                print(f"No data for battery '{battery_type}', current '{current_val}', temperature '{temp_val}'. Skipping this group.")
                continue
            
            # Interpolate missing values in 'final_voltage' (linear interpolation)
            # Even if there are no missing values, this returns the same series.
            interpolated = df_temp['final_voltage'].interpolate(method='linear')
            
            # Determine window_length for smoothing based on available samples
            n_samples = len(df_temp)
            window_length = default_window_length
            if n_samples < window_length:
                window_length = n_samples if n_samples % 2 == 1 else n_samples - 1
                if window_length < 3:
                    window_length = 3
            
            # Apply Savitzky–Golay filter on the interpolated values
            try:
                smoothed = savgol_filter(interpolated.values, window_length=window_length, polyorder=default_polyorder)
            except Exception as e:
                print(f"Error smoothing data for {battery_type}, current {current_val}, temperature {temp_val}: {e}")
                smoothed = interpolated.values
            
            # Plot smoothed values on left subplot (thin lines)
            axs[0].plot(df_temp['soc'], smoothed, marker='o', linewidth=1, label=f"{temp_val}")
            
            # Plot interpolated values on right subplot (thin lines)
            axs[1].plot(df_temp['soc'], interpolated, marker='o', linewidth=1, label=f"{temp_val}")
        
        # Configure left subplot (Smoothed Data)
        axs[0].set_title(f"{battery_type} - {current_val} (Smoothed)", fontsize=14)
        axs[0].set_xlabel("State of Charge (SoC)", fontsize=12)
        axs[0].set_ylabel("Final Voltage", fontsize=12)
        axs[0].legend(title="Temperature")
        axs[0].grid(True, color='grey', linestyle='--', linewidth=0.5)
        
        # Configure right subplot (Interpolated Data)
        axs[1].set_title(f"{battery_type} - {current_val} (Interpolated)", fontsize=14)
        axs[1].set_xlabel("State of Charge (SoC)", fontsize=12)
        axs[1].set_ylabel("Final Voltage", fontsize=12)
        axs[1].legend(title="Temperature")
        axs[1].grid(True, color='grey', linestyle='--', linewidth=0.5)
        
        # Overall figure title and tight layout
        plt.suptitle(f"{battery_type} Battery - {current_val}: Final Voltage vs. SoC\n(Smoothed vs. Interpolated)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Define filename and save the figure as PNG with tight bounding box
        filename = f"{battery_type}_{current_val}_smoothed_vs_interpolated.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, format='png', bbox_inches='tight')
        plt.close()
        
        print(f"Saved interpolation plot for battery '{battery_type}', current '{current_val}' as: {filepath}")
        graph_count += 1

if graph_count == 0:
    print("No interpolation graphs were created. Please check your data and column names.")
else:
    print(f"Total interpolation graphs created and saved: {graph_count}")
