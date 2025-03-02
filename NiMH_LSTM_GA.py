import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

from deap import base, creator, tools, algorithms

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

output_ga_folder = os.path.join('basic_model_stats', 'tuned', 'GA', 'NiMH')
os.makedirs(output_ga_folder, exist_ok=True)

print("Loading normalized NiMH data...", flush=True)
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

print(f"NiMH GA Tuner: Number of experiments = {num_experiments}", flush=True)
print(f"NiMH GA Tuner: Maximum sequence length = {max_len}", flush=True)

def eval_individual(individual):
    units = int(individual[0])
    dropout_rate = individual[1]
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(max_len, X_train.shape[2])))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=2,
                        validation_data=(X_val, y_val))
    val_loss = history.history['val_loss'][-1]
    tf.keras.backend.clear_session()
    return (val_loss,)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_units", random.randint, 32, 128)
toolbox.register("attr_dropout", random.uniform, 0.0, 0.99)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_units, toolbox.attr_dropout), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_individual)

def custom_mutate(individual, mu, sigma, indpb):
    individual, = tools.mutGaussian(individual, mu, sigma, indpb)
    individual[1] = max(0.0, min(individual[1], 0.99))
    return (individual,)

toolbox.register("mutate", custom_mutate, mu=0, sigma=10, indpb=0.2)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=10)
NGEN = 5
ga_results = []
for gen in range(NGEN):
    print(f"GA Generation {gen}", flush=True)
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
        ga_results.append({
            'generation': gen,
            'units': ind[0],
            'dropout_rate': ind[1],
            'val_loss': fit[0]
        })
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, k=1)[0]
print("Best individual (GA) for NiMH:", best_ind, "with fitness:", best_ind.fitness.values, flush=True)

ga_results_df = pd.DataFrame(ga_results)
ga_csv_path = os.path.join(output_ga_folder, 'NiMH_GA_results.csv')
ga_results_df.to_csv(ga_csv_path, index=False)
print("Saved NiMH GA tuning results to:", ga_csv_path, flush=True)

plt.figure(figsize=(10, 6))
plt.bar(ga_results_df.index, ga_results_df['val_loss'], color='salmon')
plt.xlabel('Trial Index', fontsize=12)
plt.ylabel('Validation Loss (MSE)', fontsize=12)
plt.title('NiMH GA Tuning Results', fontsize=14)
plt.tight_layout()
ga_barplot_path = os.path.join(output_ga_folder, 'NiMH_GA_barplot.png')
plt.savefig(ga_barplot_path, bbox_inches='tight')
plt.close()
print("Saved NiMH GA tuning bar plot at:", ga_barplot_path, flush=True)

best_units = int(best_ind[0])
best_dropout = best_ind[1]
print("Training final NiMH model with GA best parameters: units =", best_units, "dropout_rate =", best_dropout, flush=True)

final_model = Sequential()
final_model.add(Masking(mask_value=0.0, input_shape=(max_len, X_train.shape[2])))
final_model.add(LSTM(best_units, return_sequences=True))
final_model.add(Dropout(best_dropout))
final_model.add(Dense(1, activation='linear'))
final_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

final_early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
final_history = final_model.fit(X_train, y_train, epochs=50, batch_size=4,
                               validation_data=(X_val, y_val), callbacks=[final_early_stop])

plt.figure(figsize=(8, 5))
plt.plot(final_history.history['loss'], label='Train Loss (MSE)')
plt.plot(final_history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs. Validation Loss - NiMH Final Model (GA)', fontsize=14)
plt.legend()
plt.tight_layout()
final_loss_plot_path = os.path.join(output_ga_folder, 'NiMH_final_model_loss.png')
plt.savefig(final_loss_plot_path, bbox_inches='tight')
plt.close()
print("Saved NiMH final model loss plot at:", final_loss_plot_path, flush=True)

y_pred = final_model.predict(X_val)
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

print("NiMH Final Model (GA) Validation Metrics:", flush=True)
print("Average MSE:", avg_mse, flush=True)
print("RMSE:", rmse, flush=True)
print("Average MAE:", avg_mae, flush=True)

final_metrics_df = pd.DataFrame({'experiment': exp_val, 'MSE': mse_list, 'MAE': mae_list})
final_metrics_df.loc['Average'] = ['Average', avg_mse, avg_mae]
final_metrics_csv_path = os.path.join(output_ga_folder, 'NiMH_final_model_validation_metrics.csv')
final_metrics_df.to_csv(final_metrics_csv_path, index=False)
print("Saved NiMH final model validation metrics at:", final_metrics_csv_path, flush=True)
