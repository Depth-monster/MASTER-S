import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# --- 1. Получаем абсолютный путь к директории скрипта ---
script_dir = os.getcwd()

# --- 2. Формируем абсолютные пути к данным ---
gru_path = os.path.abspath(os.path.join(script_dir, "results/gru/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle"))
lstm_path = os.path.abspath(os.path.join(script_dir, "results/lstm/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle"))

# --- 3. Загружаем данные ---
with open(gru_path, 'rb') as f:
    gru_data = pickle.load(f)
with open(lstm_path, 'rb') as f:
    lstm_data = pickle.load(f)

# --- 4. Обрабатываем: берём только Step Ahead = 1 (первый шаг предсказания) ---
gru_pred = gru_data['pred'][1][:, 0, 0]
lstm_pred = lstm_data['pred'][1][:, 0, 0]
real_gru = gru_data['real'][1][:, 0, 0]
real_lstm = lstm_data['real'][1][:, 0, 0]

# --- 5. Построение графиков ---
N = 1200
x = np.arange(N)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# LSTM vs Real
axs[0].plot(x, real_lstm[:N], label='Real (LSTM)', color='blue')
axs[0].plot(x, lstm_pred[:N], label='LSTM Prediction', color='red')
axs[0].set_title('LSTM vs Real Values')
axs[0].legend()
axs[0].grid(True)

# GRU vs Real
axs[1].plot(x, real_gru[:N], label='Real (GRU)', color='blue')
axs[1].plot(x, gru_pred[:N], label='GRU Prediction', color='red')
axs[1].set_title('GRU vs Real Values')
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Step Ahead = 1: GRU and LSTM Predictions vs Real Values")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Сохраняем
output_dir = os.path.join(script_dir, "output_plots")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "gru_lstm_vs_real_separate.png")
plt.savefig(output_path, dpi=300)

output_path
