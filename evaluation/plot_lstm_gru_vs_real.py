import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Получаем абсолютный путь к директории скрипта ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 2. Формируем абсолютные пути к данным ---
gru_path = os.path.join(script_dir, "../results/gru/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle")
lstm_path = os.path.join(script_dir, "../results/lstm/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle")

# Преобразуем в абсолютные пути (убираем '..' и нормализуем)
gru_path = os.path.abspath(gru_path)
lstm_path = os.path.abspath(lstm_path)

# --- 3. Проверяем пути перед загрузкой ---
print("GRU Path:", gru_path)
print("LSTM Path:", lstm_path)
print("Существует GRU файл?", os.path.exists(gru_path))
print("Существует LSTM файл?", os.path.exists(lstm_path))

# --- 4. Загружаем данные (теперь с проверкой) ---
try:
    with open(gru_path, 'rb') as f:
        gru_data = pickle.load(f)
    with open(lstm_path, 'rb') as f:
        lstm_data = pickle.load(f)
except FileNotFoundError as e:
    print(f"❌ Ошибка: {e}")
    print("Проверьте пути к файлам!")
    exit(1)

# --- 5. Остальной код без изменений ---
gru_pred = np.concatenate(gru_data['pred']).flatten()
lstm_pred = np.concatenate(lstm_data['pred']).flatten()
real = np.concatenate(gru_data['real']).flatten()

N = 1000
x = np.arange(N)

plt.figure(figsize=(14, 6))
plt.plot(x, real[:N], label="Реальные значения", color="black", linewidth=2)
plt.plot(x, lstm_pred[:N], label="Прогноз LSTM", linestyle="--", color="orange")
plt.plot(x, gru_pred[:N], label="Прогноз GRU", linestyle=":", color="green")
plt.xlabel("Шаг предсказания")
plt.ylabel("Объем сетевого трафика")
plt.title("Сравнение прогноза LSTM и GRU с реальными значениями")
plt.legend()
plt.grid(True)

# Создаем папку для графиков (абсолютный путь)
output_dir = os.path.join(script_dir, "../output_plots")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "lstm_gru_vs_real.png")
plt.savefig(output_path)
print(f"✅ График сохранён в {output_path}")
