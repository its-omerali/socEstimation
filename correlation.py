import os
import seaborn as sns
import matplotlib.pyplot as plt

# Import the previous processing module. It is assumed that the file 'process_battery_data.py'
# defines a dictionary named 'battery_dfs' that maps battery types (e.g., 'LiIon', 'LiPo', 'NiMH')
# to their corresponding concatenated DataFrames.
import filemanagement as process_battery_data

# Access the dictionary of DataFrames by battery type
battery_dfs = process_battery_data.battery_dfs

# Define the folder where we will save the correlation plots (EPS and PNG)
stats_folder = os.path.join('raw_stats', 'graphs', 'correlation_plots')
os.makedirs(stats_folder, exist_ok=True)

# Define the list of columns for which we want to compute the correlation matrix.
# Ensure that these columns exist in your DataFrames.
columns_to_plot = ['capacity', 'soc', 'current', 'final_voltage']

# Iterate over each battery type and its corresponding DataFrame
for battery_type, df in battery_dfs.items():
    # Check if all required columns exist in the DataFrame.
    missing_cols = [col for col in columns_to_plot if col not in df.columns]
    if missing_cols:
        print(f"Skipping battery type '{battery_type}' due to missing columns: {missing_cols}")
        continue  # Skip this DataFrame if necessary columns are missing
    
    # Calculate the correlation matrix for the specified columns.
    corr_matrix = df[columns_to_plot].corr()
    
    # Create a new figure for the correlation plot with a specified figure size.
    plt.figure(figsize=(8, 6))
    
    # Generate the heatmap using seaborn, with annotations and a chosen color map.
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title(f"Correlation Matrix for {battery_type} Battery")
    
    # Define the base filename for saving the plots.
    filename_base = os.path.join(stats_folder, f"{battery_type}_correlation")
    
    # Save the plot in EPS format with tight bounding box to remove extra whitespace.
    plt.savefig(filename_base + ".eps", format='eps', bbox_inches='tight')
    # Save the plot in PNG format similarly.
    plt.savefig(filename_base + ".png", format='png', bbox_inches='tight')
    
    # Close the current figure to free up memory.
    plt.close()
    
    print(f"Saved correlation plots for '{battery_type}' as EPS and PNG in '{stats_folder}' folder.")
