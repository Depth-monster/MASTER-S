import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Функция объединения предсказаний из окон с усреднением
def merge_overlapping_predictions(predictions, look_ahead):
    total_len = len(predictions) + look_ahead - 1
    merged = np.zeros(total_len)
    count = np.zeros(total_len)

    for i, window in enumerate(predictions):
        for j in range(min(look_ahead, len(window))):
            merged[i + j] += window[j]
            count[i + j] += 1

    return merged / np.where(count == 0, 1, count)

# Путь к проекту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

gru_path = os.path.join(ROOT_DIR, 'results', 'gru', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')
lstm_path = os.path.join(ROOT_DIR, 'results', 'lstm', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')

# Загрузка данных
with open(gru_path, 'rb') as f:
    data_gru = pickle.load(f)

with open(lstm_path, 'rb') as f:
    data_lstm = pickle.load(f)

# Объединяем окна
look_ahead = 20
y_true = merge_overlapping_predictions(data_gru['real'], look_ahead)
y_pred_gru = merge_overlapping_predictions(data_gru['pred'], look_ahead)
y_pred_lstm = merge_overlapping_predictions(data_lstm['pred'], look_ahead)

# Сглаживание (по желанию)
def smooth(y, window=5):
    return np.convolve(y, np.ones(window)/window, mode='same')

y_true_s = smooth(y_true)
y_pred_gru_s = smooth(y_pred_gru)
y_pred_lstm_s = smooth(y_pred_lstm)

# Ограничим длину для наглядности
N = 300
x = np.arange(N)

# Построение графика
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# GRU
axs[0].plot(x, y_true_s[:N], label='Real Values', color='blue')
axs[0].plot(x, y_pred_gru_s[:N], label='GRU Prediction', color='red')
axs[0].set_title('Prediction Result of GRU Model')
axs[0].legend()
axs[0].grid(True)

# LSTM
axs[1].plot(x, y_true_s[:N], label='Real Values', color='blue')
axs[1].plot(x, y_pred_lstm_s[:N], label='LSTM Prediction', color='red')
axs[1].set_title('Prediction Result of LSTM Model')
axs[1].legend()
axs[1].grid(True)

plt.suptitle('Figure 3.7: Difference of the Test Result Between Two RNN Variants')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('gru_lstm_smooth_comparison.png', dpi=300)
plt.show()
