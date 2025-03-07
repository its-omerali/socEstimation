# THIS FILE INPUTS BEST PARAMETERS OF GA CALCULATED IN PREVIOUS STEPS
# AND USES THEM TO BUILD AND FIT THE MODEL AND THEN PLOTS ALL THE VALUES
# THIS WORKS FINE TO VALIDATE THAT THE PARAMETERS ARE CORRECT AND COULD BE USED
# IN OTHER MODELS. HOWEVER, ONE COMMON GRAPH MAY NOT BE SUITABLE FOR ACADEMIC PURPOSES
# THEREFORE, ITS EXTENSION IN THE FORM OF LiIon_LSTM_BestFit_A.PY WILL RESOLVE THESE ISSUES
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
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
if 'current_norm' in df.columns:
    current_col = 'current_norm'

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
    experiment_ids.append(f"LiIon_{curr}_{temp}")

max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
y_padded = pad_sequences(y_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
print("Prepared data: max sequence length =", max_len, flush=True)

num_experiments = len(X_padded)
split_index = int(0.8 * num_experiments)
X_train, X_val = X_padded[:split_index], X_padded[split_index:]
y_train, y_val = y_padded[:split_index], y_padded[split_index:]
exp_train = experiment_ids[:split_index]
exp_val = experiment_ids[split_index:]
print("Number of experiments:", num_experiments, flush=True)

# -------------------------
# 3. Load Best GA Parameters for LiIon
# -------------------------
# (Assuming GA tuning results CSV is stored under basic_model_stats/tuned/GA/LiIon)
ga_csv_path = os.path.join('basic_model_stats', 'tuned', 'GA', 'LiIon', 'LiIon_GA_results.csv')
ga_results_df = pd.read_csv(ga_csv_path)
# Choose best parameters: the row with the minimum val_loss
best_ga = ga_results_df.loc[ga_results_df['val_loss'].idxmin()]
best_units = int(best_ga['units'])
best_dropout = float(best_ga['dropout_rate'])
print(f"Best GA parameters for LiIon: units = {best_units}, dropout_rate = {best_dropout}", flush=True)

# -------------------------
# 4. Build and Train Final Model Using GA Parameters
# -------------------------
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(max_len, X_train.shape[2])))
model.add(LSTM(best_units, return_sequences=True))
model.add(Dropout(best_dropout))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Training final model with GA parameters...", flush=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=4,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop], verbose=1)

# -------------------------
# 5. Evaluate Regression Metrics on Validation Set
# -------------------------
y_pred = model.predict(X_val)
mse_list = []
mae_list = []
for i in range(len(y_val)):
    # Identify non-padded rows (assume padded rows are all zeros)
    non_padded_idx = np.where(np.any(X_val[i] != 0.0, axis=1))[0]
    if len(non_padded_idx) == 0:
        continue
    last_idx = non_padded_idx[-1] + 1
    mse = mean_squared_error(y_val[i][:last_idx], y_pred[i][:last_idx])
    mae = mean_absolute_error(y_val[i][:last_idx], y_pred[i][:last_idx])
    mse_list.append(mse)
    mae_list.append(mae)
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)
rmse = np.sqrt(avg_mse)

metrics_dict = {
    'Average MSE': avg_mse,
    'RMSE': rmse,
    'Average MAE': avg_mae
}
metrics_df = pd.DataFrame([metrics_dict])
metrics_csv_path = os.path.join(output_final_folder, 'final_metrics_GA.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print("Saved final regression metrics (GA) at:", metrics_csv_path, flush=True)

# -------------------------
# 6. Plot Actual vs. Predicted Discharge Curves
# -------------------------
# We plot, for each experiment in the validation set, a line for actual and predicted discharge curve.
# X-axis: SoC (actual or predicted), Y-axis: final_voltage_norm (from input features, index 1)
plt.figure(figsize=(10, 6))
for i in range(len(y_val)):
    # Find non-padded indices
    non_padded_idx = np.where(np.any(X_val[i] != 0.0, axis=1))[0]
    if len(non_padded_idx) == 0:
        continue
    last_idx = non_padded_idx[-1] + 1
    # Actual: use actual soc values (target) and final_voltage_norm from input
    actual_soc = y_val[i][:last_idx].flatten()
    voltage = X_val[i][:last_idx, 1].flatten()  # final_voltage_norm is feature index 1
    # Predicted: predicted soc values and same voltage
    predicted_soc = y_pred[i][:last_idx].flatten()
    plt.plot(actual_soc, voltage, 'b-', alpha=0.3, label='Actual' if i==0 else "")
    plt.plot(predicted_soc, voltage, 'r--', alpha=0.3, label='Predicted' if i==0 else "")
plt.xlabel('SoC (normalized)', fontsize=12)
plt.ylabel('Final Voltage (normalized)', fontsize=12)
plt.title('LiIon Discharge Curve: Actual vs Predicted (GA Model)', fontsize=14)
plt.legend()
plt.tight_layout()
discharge_plot_path = os.path.join(output_final_folder, 'discharge_curve_GA.png')
plt.savefig(discharge_plot_path, bbox_inches='tight')
plt.close()
print("Saved LiIon discharge curve plot (GA model) at:", discharge_plot_path, flush=True)
