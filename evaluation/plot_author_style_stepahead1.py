import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Путь к проекту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Пути к данным
gru_path = os.path.join(ROOT_DIR, 'results', 'gru', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')
lstm_path = os.path.join(ROOT_DIR, 'results', 'lstm', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')

# Загрузка
with open(gru_path, 'rb') as f:
    data_gru = pickle.load(f)

with open(lstm_path, 'rb') as f:
    data_lstm = pickle.load(f)

# Берём только тестовую часть: [1]
# Извлекаем только первый шаг предсказания (t+1)
y_true_gru = data_gru['real'][1][:, 0, 0]
y_pred_gru = data_gru['pred'][1][:, 0, 0]
y_true_lstm = data_lstm['real'][1][:, 0, 0]
y_pred_lstm = data_lstm['pred'][1][:, 0, 0]

# Обрезаем для визуализации
N = 300
x = np.arange(N)

# Построение графиков
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# GRU график
axs[0].plot(x, y_true_gru[:N], label='Real Values', color='blue')
axs[0].plot(x, y_pred_gru[:N], label='GRU Prediction', color='red')
axs[0].set_title('Prediction Result of GRU Model')
axs[0].legend()
axs[0].grid(True)

# LSTM график
axs[1].plot(x, y_true_lstm[:N], label='Real Values', color='blue')
axs[1].plot(x, y_pred_lstm[:N], label='LSTM Prediction', color='red')
axs[1].set_title('Prediction Result of LSTM Model')
axs[1].legend()
axs[1].grid(True)

plt.suptitle('Figure 3.7: Difference of the Test Result Between Two RNN Variants (Step Ahead = 1)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('stepahead1_author_style.png', dpi=300)
plt.show()
