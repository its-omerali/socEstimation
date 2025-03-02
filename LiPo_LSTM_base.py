# LiPo_LSTM.ipynb
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

# Set output folder for LiPo model stats
output_stats_folder = os.path.join('basic_model_stats', 'untuned', 'LiPo')
os.makedirs(output_stats_folder, exist_ok=True)

# Load normalized LiPo pickle file
with open('data/normalized/LiPo_normalized.pkl', 'rb') as f:
    df = pickle.load(f)

# Determine current column
if 'current' in df.columns:
    current_col = 'current'
elif 'currents' in df.columns:
    current_col = 'currents'
else:
    raise ValueError("No current column found in LiPo dataframe.")

# Group data by current and temperature
grouped = df.groupby([current_col, 'temperature'])

X_sequences = []
y_sequences = []
experiment_ids = []

for (curr, temp), group_df in grouped:
    group_df = group_df.sort_values('timestamp_num_norm')
    X = group_df[['timestamp_num_norm', 'final_voltage_norm', 'current_norm', 'capacity_norm']].values
    y = group_df[['soc_norm']].values
    X_sequences.append(X)
    y_sequences.append(y)
    experiment_ids.append(f"LiPo_{curr}_{temp}")

max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
y_padded = pad_sequences(y_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)

num_experiments = len(X_padded)
split_index = int(0.8 * num_experiments)
X_train, X_val = X_padded[:split_index], X_padded[split_index:]
y_train, y_val = y_padded[:split_index], y_padded[split_index:]
exp_train = experiment_ids[:split_index]
exp_val = experiment_ids[split_index:]

print("LiPo: Number of experiments:", num_experiments)
print("LiPo: Maximum sequence length:", max_len)

input_shape = (max_len, X_train.shape[2])
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=input_shape))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
csv_logger = CSVLogger(os.path.join(output_stats_folder, 'training_log.csv'), append=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_val, y_val),
                    callbacks=[early_stop, csv_logger])

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss - LiPo')
plt.legend()
plt.tight_layout()
loss_plot_path = os.path.join(output_stats_folder, 'loss_plot.png')
plt.savefig(loss_plot_path, bbox_inches='tight')
plt.close()
print(f"Saved LiPo loss plot at: {loss_plot_path}")

y_pred = model.predict(X_val)
mse_list = []
mae_list = []
for i in range(len(y_val)):
    seq = X_val[i]
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

print("LiPo Validation Metrics:")
print("Average MSE:", avg_mse)
print("RMSE:", rmse)
print("Average MAE:", avg_mae)

metrics_df = pd.DataFrame({
    'experiment': exp_val,
    'MSE': mse_list,
    'MAE': mae_list
})
metrics_df.loc['Average'] = ['Average', avg_mse, avg_mae]
metrics_csv_path = os.path.join(output_stats_folder, 'validation_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved LiPo validation metrics at: {metrics_csv_path}")
