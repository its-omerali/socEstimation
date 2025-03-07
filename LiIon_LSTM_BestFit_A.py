import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# 1. Set Up Output Directory
# -------------------------
output_final_folder = os.path.join('socEstimation', 'final_analysis', 'LiIon')
os.makedirs(output_final_folder, exist_ok=True)
print("Output folder for final analysis:", output_final_folder, flush=True)

# -------------------------
# 2. Load Normalized Data and Prepare Experiments
# -------------------------
with open('data/normalized/LiIon_normalized.pkl', 'rb') as f:
    df = pickle.load(f)
    
# Determine current column (prefer 'current' then 'currents')
if 'current' in df.columns:
    current_col = 'current'

else:
    raise ValueError("No current column found in LiIon dataframe.")

# Group data by (current, temperature); each group is one experiment
grouped = df.groupby([current_col, 'temperature'])
X_sequences, y_sequences, experiment_ids = [], [], []
for (curr, temp), group_df in grouped:
    group_df = group_df.sort_values('timestamp_num_norm')
    # Input features: [timestamp_num_norm, final_voltage_norm, current_norm, capacity_norm]
    X = group_df[['timestamp_num_norm', 'final_voltage_norm', 'current_norm', 'capacity_norm']].values
    # Target: [soc_norm]
    y = group_df[['soc_norm']].values
    X_sequences.append(X)
    y_sequences.append(y)
    experiment_ids.append(f"{curr}_{temp}")  # e.g. "30mA_5C"

max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
y_padded = pad_sequences(y_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
print("Prepared data: max sequence length =", max_len, flush=True)

num_experiments = len(X_padded)
split_index = int(0.8 * num_experiments)
X_train, X_val = X_padded[:split_index], X_padded[split_index:]
y_train, y_val = y_padded[:split_index], y_padded[split_index:]
exp_val = experiment_ids[split_index:]
print("Number of experiments (validation):", len(X_val), flush=True)

# -------------------------
# 3. Load Best GA Parameters (assumed stored in a CSV)
# -------------------------
ga_csv_path = os.path.join('basic_model_stats', 'tuned', 'GA', 'LiIon', 'LiIon_GA_results.csv')
ga_results_df = pd.read_csv(ga_csv_path)
# Choose best parameters (lowest val_loss)
best_ga = ga_results_df.loc[ga_results_df['val_loss'].idxmin()]
best_units = int(best_ga['units'])
best_dropout = float(best_ga['dropout_rate'])
print(f"Best GA parameters for LiIon: units = {best_units}, dropout_rate = {best_dropout}", flush=True)

# -------------------------
# 4. Build the Final Model and Predict
# -------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(max_len, X_train.shape[2])))
model.add(LSTM(best_units, return_sequences=True))
model.add(Dropout(best_dropout))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# (Assume the model is already trained; if not, you can load weights or retrain using the same training data.)
# Here, for final analysis, we predict on the validation set:
y_pred = model.predict(X_val)

# -------------------------
# 5. Evaluate Regression Metrics for Each Experiment
# -------------------------
results = []
for i, exp_id in enumerate(exp_val):
    seq = X_val[i]
    non_padded = np.where(np.any(seq != 0.0, axis=1))[0]
    if len(non_padded) == 0:
        continue
    last_idx = non_padded[-1] + 1
    true_vals = y_val[i][:last_idx]
    pred_vals = y_pred[i][:last_idx]
    mse = mean_squared_error(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    # exp_id is like "30mA_5C"
    current_val, temperature_val = exp_id.split('_')
    results.append({
        'experiment': exp_id,
        'current': current_val,
        'temperature': temperature_val,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    })
metrics_df = pd.DataFrame(results)
metrics_csv_path = os.path.join(output_final_folder, 'final_regression_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print("Saved final regression metrics at:", metrics_csv_path, flush=True)

# -------------------------
# 6. Plot Actual vs. Predicted Discharge Curves per Current
# -------------------------
# We'll group experiments by current (e.g., "30mA", "50mA", "100mA")
current_groups = {}
for i, exp_id in enumerate(exp_val):
    current_val, temperature_val = exp_id.split('_')
    if current_val not in current_groups:
        current_groups[current_val] = []
    current_groups[current_val].append((temperature_val, X_val[i], y_val[i], y_pred[i]))

# Define a colormap for temperatures (one color per temperature)
import matplotlib.cm as cm
temperatures_all = sorted(metrics_df['temperature'].unique(), key=lambda x: int(x.rstrip('C')))
colors = {temp: cm.viridis(i/len(temperatures_all)) for i, temp in enumerate(temperatures_all)}

for current_val, experiments in current_groups.items():
    plt.figure(figsize=(10, 6))
    for (temp, X_seq, y_true_seq, y_pred_seq) in experiments:
        non_padded = np.where(np.any(X_seq != 0.0, axis=1))[0]
        if len(non_padded) == 0:
            continue
        last_idx = non_padded[-1] + 1
        # For discharge curves, we use SoC (target) as x-axis and final_voltage_norm (feature index 1) as y-axis.
        actual_soc = y_true_seq[:last_idx].flatten()
        predicted_soc = y_pred_seq[:last_idx].flatten()
        voltage = X_seq[:last_idx, 1].flatten()
        # Plot actual (solid) and predicted (dashed) using the same color.
        plt.plot(actual_soc, voltage, color=colors[temp], linestyle='-', linewidth=2,
                 label=f"Actual {temp}" if current_val+"_"+temp not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot(predicted_soc, voltage, color=colors[temp], linestyle='--', linewidth=2,
                 label=f"Predicted {temp}" if current_val+"_"+temp not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.xlabel("SoC (normalized)", fontsize=12)
    plt.ylabel("Final Voltage (normalized)", fontsize=12)
    plt.title(f"LiIon Discharge Curve at {current_val}", fontsize=14)
    # Add both major and minor grids in light grey
    plt.grid(which='major', linestyle='-', linewidth=0.75, color='lightgrey')
    plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_final_folder, f"discharge_curve_{current_val}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved LiIon discharge curve plot for {current_val} at: {plot_path}", flush=True)

# -------------------------
# 7. Plot Bar Charts for Metrics per Current
# -------------------------
# For each current, create a bar chart of MSE, RMSE, MAE (averaged over temperatures)
metric_plots = {}
for current_val in current_groups:
    df_current = metrics_df[metrics_df['current'] == current_val]
    # Average metrics for each temperature
    avg_metrics = df_current.groupby('temperature').agg({'MSE': 'mean', 'RMSE': 'mean', 'MAE': 'mean'}).reset_index()
    # Plot each metric as a grouped bar chart
    x = np.arange(len(avg_metrics))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, avg_metrics['MSE'], width, label='MSE', color='skyblue')
    plt.bar(x, avg_metrics['RMSE'], width, label='RMSE', color='salmon')
    plt.bar(x + width, avg_metrics['MAE'], width, label='MAE', color='limegreen')
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.title(f"LiIon Regression Metrics at {current_val}", fontsize=14)
    plt.xticks(x, avg_metrics['temperature'])
    plt.legend()
    plt.grid(which='major', linestyle='-', linewidth=0.75, color='lightgrey')
    plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgrey')
    plt.minorticks_on()
    plt.tight_layout()
    bar_plot_path = os.path.join(output_final_folder, f"metrics_bar_{current_val}.png")
    plt.savefig(bar_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved LiIon metrics bar plot for {current_val} at: {bar_plot_path}", flush=True)
