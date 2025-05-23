import pandas as pd
import matplotlib.pyplot as plt
import os

# Значения MAPE по шагам для GRU и LSTM
mape_gru = [
    4.42, 5.19, 5.52, 5.79, 6.06, 6.33, 6.56, 6.72, 6.88, 7.06,
    7.23, 7.38, 7.54, 7.71, 7.81, 7.95, 8.12, 8.27, 8.48, 8.63
]

mape_lstm = [
    13.86, 10.63, 9.80, 8.99, 8.95, 9.22, 9.54, 9.85, 10.11, 10.32,
    10.50, 10.65, 10.80, 10.95, 11.08, 11.20, 11.31, 11.42, 11.53, 11.62
]

steps = list(range(1, 21))

# --- Построение графика ---
plt.figure(figsize=(12, 6))
plt.plot(steps, mape_gru, label="GRU", color='blue')
plt.plot(steps, mape_lstm, label="LSTM", color='red')

plt.xlabel("Шаг предсказания", fontsize=12)
plt.ylabel("MAPE (средняя абсолютная процентная ошибка, %)", fontsize=12)
plt.title("Сравнение ошибок MAPE моделей GRU и LSTM", fontsize=14)
plt.legend()
plt.grid(True)

# Сохранение
output_path = os.path.join("output_plots", "mape_comparison_gru_lstm.png")
os.makedirs("output_plots", exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"✅ График сохранён: {output_path}")
