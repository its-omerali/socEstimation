# LiIon_LSTM.ipynb
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set output folder for LiIon model stats and create it if it doesn't exist
output_stats_folder = os.path.join('basic_model_stats', 'untuned', 'LiIon')
os.makedirs(output_stats_folder, exist_ok=True)

# Load normalized LiIon pickle file
with open('data/normalized/LiIon_normalized.pkl', 'rb') as f:
    df = pickle.load(f)

# Now the column current corresponds to the actual current measured with load attached. 
# whereas currents (with an s) is the set current value for each experiment type (such as 30mA, 50mA, 100mA)
# I will later change (currents) to (experiment_currents) to avoid confusion.

if 'current' in df.columns:
    current_col = 'current'
else:
    raise ValueError("No current column found in LiIon dataframe.")

# Group data by current and temperature (each group is one experiment)
grouped = df.groupby([current_col, 'temperature'])

# Prepare lists for sequences and experiment identifiers
X_sequences = []
y_sequences = []
experiment_ids = []

for (curr, temp), group_df in grouped:
    group_df = group_df.sort_values('timestamp_num_norm')
    # Input features and target from normalized columns
    X = group_df[['timestamp_num_norm', 'final_voltage_norm', 'current_norm', 'capacity_norm']].values
    y = group_df[['soc_norm']].values  # target shape: (n_timesteps, 1)
    X_sequences.append(X)
    y_sequences.append(y)
    experiment_ids.append(f"LiIon_{curr}_{temp}")

# Pad sequences to the maximum length across experiments (padding at the end)
max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
y_padded = pad_sequences(y_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)

# Split into training (80%) and validation (20%) sets (split at experiment level)
num_experiments = len(X_padded)
split_index = int(0.8 * num_experiments)
X_train, X_val = X_padded[:split_index], X_padded[split_index:]
y_train, y_val = y_padded[:split_index], y_padded[split_index:]
exp_train = experiment_ids[:split_index]
exp_val = experiment_ids[split_index:]

print("LiIon: Number of experiments:", num_experiments)
print("LiIon: Maximum sequence length:", max_len)

# Build the Keras LSTM model
input_shape = (max_len, X_train.shape[2])  # (timesteps, features)
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=input_shape))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Set up callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
csv_logger = CSVLogger(os.path.join(output_stats_folder, 'training_log.csv'), append=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_val, y_val),
                    callbacks=[early_stop, csv_logger])

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss - LiIon')
plt.legend()
plt.tight_layout()
loss_plot_path = os.path.join(output_stats_folder, 'loss_plot.png')
plt.savefig(loss_plot_path, bbox_inches='tight')
plt.close()
print(f"Saved LiIon loss plot at: {loss_plot_path}")

# Evaluate model on validation set by computing MSE, RMSE, MAE per experiment
y_pred = model.predict(X_val)
mse_list = []
mae_list = []
for i in range(len(y_val)):
    seq = X_val[i]
    # Identify non-padded time steps (assume padded rows are all zeros)
    non_padded_idx = np.where(np.any(seq != 0.0, axis=1))[0]
    if len(non_padded_idx) == 0:
        continue
    last_idx = non_padded_idx[-1] + 1
    true_vals = y_val[i][:last_idx]
    pred_vals = y_pred[i][:last_idx]
    mse = mean_squared_error(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    mse_list.append(mse)
    mae_list.append(mae)

avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)
rmse = np.sqrt(avg_mse)

print("LiIon Validation Metrics:")
print("Average MSE:", avg_mse)
print("RMSE:", rmse)
print("Average MAE:", avg_mae)

# Save validation metrics to CSV
metrics_df = pd.DataFrame({
    'experiment': exp_val,
    'MSE': mse_list,
    'MAE': mae_list
})
metrics_df.loc['Average'] = ['Average', avg_mse, avg_mae]
metrics_csv_path = os.path.join(output_stats_folder, 'validation_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved LiIon validation metrics at: {metrics_csv_path}")
