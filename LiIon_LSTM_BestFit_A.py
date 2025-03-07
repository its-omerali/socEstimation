#THIS IS AN EXTENTION TO LIION_LSTM_BESTFIT.PY AND IMPROVES THE CODE BY MAKING SEPARATE PLOTS
# FOR EACH CORRESPONDING TEMPERATURE AND LOAD VALUES
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ensure immediate flush of prints
print("Starting combined final analysis for LiIon...", flush=True)

# -------------------------
# 1. Set Up Output Directory
# -------------------------
results_folder = os.path.join('socEstimation', 'final_analysis', 'LiIon', 'Results')
os.makedirs(results_folder, exist_ok=True)
print("Results will be stored in:", results_folder, flush=True)

# -------------------------
# 2. Load Normalized Data and Prepare Validation Experiments
# -------------------------
with open('data/normalized/LiIon_normalized.pkl', 'rb') as f:
    df = pickle.load(f)
    
# Determine current column
if 'currents' in df.columns:
    current_col = 'currents'
else:
    raise ValueError("No current column found in LiIon dataframe.")

# Group data by (current, temperature)
grouped = df.groupby([current_col, 'temperature'])
X_sequences, y_sequences, exp_ids = [], [], []
for (curr, temp), group_df in grouped:
    group_df = group_df.sort_values('timestamp_num_norm')
    X = group_df[['timestamp_num', 'final_voltage', 'current', 'capacity']].values
    y = group_df[['soc']].values
    X_sequences.append(X)
    y_sequences.append(y)
    exp_ids.append(f"{curr}_{temp}")  # e.g., "30mA_5C"

# Pad sequences
max_len = max(len(seq) for seq in X_sequences)
X_all = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
y_all = pad_sequences(y_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
print("Max sequence length:", max_len, flush=True)

# For final analysis, we use the validation set (assume 80/20 split by experiments)
num_experiments = len(X_all)
split_index = int(0.8 * num_experiments)
X_val = X_all[split_index:]
y_val = y_all[split_index:]
exp_val = exp_ids[split_index:]
print("Number of validation experiments:", len(X_val), flush=True)

# -------------------------
# 3. Load Best GA Parameters and Build Final Model
# -------------------------
# For this example, we load the best GA parameters from CSV.
ga_csv_path = os.path.join('basic_model_stats', 'tuned', 'GA', 'LiIon', 'LiIon_GA_results.csv')
ga_results_df = pd.read_csv(ga_csv_path)
best_row = ga_results_df.loc[ga_results_df['val_loss'].idxmin()]
best_units = int(best_row['units'])
best_dropout = float(best_row['dropout_rate'])
print(f"Best GA parameters for LiIon: units={best_units}, dropout_rate={best_dropout}", flush=True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense
final_model = Sequential()
final_model.add(Masking(mask_value=0.0, input_shape=(max_len, X_val.shape[2])))
final_model.add(LSTM(best_units, return_sequences=True))
final_model.add(Dropout(best_dropout))
final_model.add(Dense(1, activation='linear'))
final_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# (Assuming final_model is already trained; if not, you can train or load weights)
# For this example, we simply predict using the current final_model.
y_pred = final_model.predict(X_val)
print("Computed predictions on validation set.", flush=True)

# -------------------------
# 4. Combine Experiments by Current
# -------------------------
# Create a dictionary: keys = current (e.g., "30mA", "50mA", "100mA"),
# values = list of tuples (temperature, X_ex, y_ex, y_pred_ex)
curr_dict = {}
for i, exp_id in enumerate(exp_val):
    curr, temp = exp_id.split('_')
    if curr not in curr_dict:
        curr_dict[curr] = []
    curr_dict[curr].append((temp, X_val[i], y_val[i], y_pred[i]))

# -------------------------
# 5. Plot Combined Actual vs Predicted for Each Current
# -------------------------
# Define a colormap to assign a unique color for each temperature.
import matplotlib.cm as cm
all_temperatures = sorted({temp for experiments in curr_dict.values() for (temp,_,_,_) in experiments}, key=lambda x: int(x.rstrip('C')))
color_map = {temp: cm.viridis(i/len(all_temperatures)) for i, temp in enumerate(all_temperatures)}

# Create a DataFrame to store regression metrics for each (current, temperature)
metrics_records = []

for curr, experiments in curr_dict.items():
    plt.figure(figsize=(10, 6))
    # For each experiment (for a given current) plot actual vs predicted.
    for (temp, X_ex, y_ex, y_pred_ex) in experiments:
        # Determine non-padded indices
        non_padded = np.where(np.any(X_ex != 0.0, axis=1))[0]
        if len(non_padded) == 0:
            continue
        last_idx = non_padded[-1] + 1
        actual_soc = y_ex[:last_idx].flatten()       # actual SoC (target)
        predicted_soc = y_pred_ex[:last_idx].flatten() # predicted SoC
        voltage = X_ex[:last_idx, 1].flatten()           # final_voltage_norm from features
        
        # Compute metrics for this (curr, temp)
        mse_val = mean_squared_error(actual_soc, predicted_soc)
        mae_val = mean_absolute_error(actual_soc, predicted_soc)
        rmse_val = np.sqrt(mse_val)
        metrics_records.append({
            'current': curr,
            'temperature': temp,
            'MSE': mse_val,
            'RMSE': rmse_val,
            'MAE': mae_val
        })
        
        # Plot actual (solid) and predicted (dashed) lines for this temperature in the same color.
        plt.plot(actual_soc, voltage, color=color_map[temp], linestyle='-', linewidth=2, label=f"Actual {temp}")
        plt.plot(predicted_soc, voltage, color=color_map[temp], linestyle='--', linewidth=2, label=f"Predicted {temp}")
    
    plt.xlabel("SoC (normalized)", fontsize=12)
    plt.ylabel("Final Voltage (normalized)", fontsize=12)
    plt.title(f"LiIon Discharge Curve at {curr}", fontsize=14)
    # Add major and minor grids in light grey
    plt.grid(which='major', linestyle='-', linewidth=0.75, color='lightgrey')
    plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')
    plt.minorticks_on()
    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plot_filename = f"LiIon_discharge_{curr}.png"
    plot_path = os.path.join(results_folder, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined discharge curve for {curr} at: {plot_path}", flush=True)

# -------------------------
# 6. Save Regression Metrics to CSV
# -------------------------
metrics_df = pd.DataFrame(metrics_records)
csv_path = os.path.join(results_folder, "final_regression_metrics_by_current_temperature.csv")
metrics_df.to_csv(csv_path, index=False)
print("Saved regression metrics CSV at:", csv_path, flush=True)

