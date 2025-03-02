# NiMH_tuner.ipynb
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
try:
    import keras_tuner as kt
except ImportError:
    import kerastuner as kt

output_tuned_folder = os.path.join('basic_model_stats', 'tuned', 'KerasTuner', 'NiMH')
os.makedirs(output_tuned_folder, exist_ok=True)

with open('data/normalized/NiMH_normalized.pkl', 'rb') as f:
    df = pickle.load(f)

if 'current_norm' in df.columns:
    current_col = 'current_norm'
else:
    raise ValueError("No current column found in NiMH dataframe.")

grouped = df.groupby([current_col, 'temperature'])
X_sequences, y_sequences, experiment_ids = [], [], []
for (curr, temp), group_df in grouped:
    group_df = group_df.sort_values('timestamp_num_norm')
    X = group_df[['timestamp_num_norm', 'final_voltage_norm', 'current_norm', 'capacity_norm']].values
    y = group_df[['soc_norm']].values
    X_sequences.append(X)
    y_sequences.append(y)
    experiment_ids.append(f"NiMH_{curr}_{temp}")

max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)
y_padded = pad_sequences(y_sequences, maxlen=max_len, dtype='float32', padding='post', value=0.0)

num_experiments = len(X_padded)
split_index = int(0.8 * num_experiments)
X_train, X_val = X_padded[:split_index], X_padded[split_index:]
y_train, y_val = y_padded[:split_index], y_padded[split_index:]
exp_train = experiment_ids[:split_index]
exp_val = experiment_ids[split_index:]

print("NiMH: Number of experiments:", num_experiments)
print("NiMH: Maximum sequence length:", max_len)

def model_builder(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(max_len, X_train.shape[2])))
    units = hp.Int('units', min_value=32, max_value=128, step=32, default=50)
    model.add(LSTM(units, return_sequences=True))
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner = kt.RandomSearch(
    model_builder,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory=output_tuned_folder,
    project_name='NiMH_tuning'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_val, y_val), callbacks=[early_stop])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters for NiMH:", best_hps.values)

trial_results = []
for trial_id, trial in tuner.oracle.trials.items():
    hp_values = trial.hyperparameters.values
    score = trial.score
    trial_results.append({
        'trial_id': trial_id,
        'units': hp_values.get('units', None),
        'dropout_rate': hp_values.get('dropout_rate', None),
        'val_loss': score
    })
trials_df = pd.DataFrame(trial_results)
best_trial = trials_df.loc[trials_df['val_loss'].idxmin()]
print("Best trial details:")
print(best_trial)
trials_csv_path = os.path.join(output_tuned_folder, 'NiMH_tuner_results.csv')
trials_df.to_csv(trials_csv_path, index=False)
print("Saved NiMH tuner results to:", trials_csv_path)

plt.figure(figsize=(10, 6))
plt.bar(trials_df['trial_id'], trials_df['val_loss'], color='skyblue')
plt.xlabel('Trial ID', fontsize=12)
plt.ylabel('Validation Loss (MSE)', fontsize=12)
plt.title('NiMH - Hyperparameter Tuning Results', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
bar_plot_path = os.path.join(output_tuned_folder, 'NiMH_tuner_barplot.png')
plt.savefig(bar_plot_path, bbox_inches='tight')
plt.close()
print("Saved NiMH tuner bar plot at:", bar_plot_path)

best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_val, y_val),
                          callbacks=[early_stop])

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss - NiMH Best Model')
plt.legend()
plt.tight_layout()
best_loss_plot_path = os.path.join(output_tuned_folder, 'NiMH_best_model_loss.png')
plt.savefig(best_loss_plot_path, bbox_inches='tight')
plt.close()
print("Saved NiMH best model loss plot at:", best_loss_plot_path)

y_pred = best_model.predict(X_val)
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

print("NiMH Best Model Validation Metrics:")
print("Average MSE:", avg_mse)
print("RMSE:", rmse)
print("Average MAE:", avg_mae)

metrics_df = pd.DataFrame({
    'experiment': exp_val,
    'MSE': mse_list,
    'MAE': mae_list
})
metrics_df.loc['Average'] = ['Average', avg_mse, avg_mae]
metrics_csv_path = os.path.join(output_tuned_folder, 'NiMH_best_model_validation_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print("Saved NiMH best model validation metrics at:", metrics_csv_path)
