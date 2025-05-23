import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Настройки ---
N = 300  # Количество точек для отображения
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "../output_plots")
os.makedirs(output_dir, exist_ok=True)

# --- Пути к данным ---
gru_path = os.path.abspath(os.path.join(script_dir, "../results/gru/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle"))
lstm_path = os.path.abspath(os.path.join(script_dir, "../results/lstm/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle"))

# --- Загрузка ---
with open(gru_path, 'rb') as f:
    gru_data = pickle.load(f)

with open(lstm_path, 'rb') as f:
    lstm_data = pickle.load(f)

# --- Обработка ---
gru_pred = np.concatenate(gru_data['pred']).flatten()
lstm_pred = np.concatenate(lstm_data['pred']).flatten()
real = np.concatenate(gru_data['real']).flatten()  # Можно взять из любого — одинаково

x = np.arange(N)

# --- GRU vs Real ---
plt.figure(figsize=(12, 4))
plt.plot(x, real[:N], label="Реальные значения", color="black", linewidth=2)
plt.plot(x, gru_pred[:N], label="Прогноз GRU", color="green", linestyle=":")
plt.xlabel("Шаг предсказания")
plt.ylabel("Объем сетевого трафика")
plt.title("GRU: Прогноз vs Реальные значения")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gru_vs_real.png"))

# --- LSTM vs Real ---
plt.figure(figsize=(12, 4))
plt.plot(x, real[:N], label="Реальные значения", color="black", linewidth=2)
plt.plot(x, lstm_pred[:N], label="Прогноз LSTM", color="orange", linestyle="--")
plt.xlabel("Шаг предсказания")
plt.ylabel("Объем сетевого трафика")
plt.title("LSTM: Прогноз vs Реальные значения")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lstm_vs_real.png"))

print("✅ Графики сохранены в папке output_plots")
