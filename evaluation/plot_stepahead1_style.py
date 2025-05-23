import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

gru_path = os.path.join(ROOT_DIR, 'results', 'gru', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')
lstm_path = os.path.join(ROOT_DIR, 'results', 'lstm', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')

with open(gru_path, 'rb') as f:
    data_gru = pickle.load(f)

with open(lstm_path, 'rb') as f:
    data_lstm = pickle.load(f)

# Step Ahead = 1 → только первый элемент в каждом окне
y_true_gru = np.array([x[0] for x in data_gru['real']])
y_pred_gru = np.array([x[0] for x in data_gru['pred']])
y_true_lstm = np.array([x[0] for x in data_lstm['real']])
y_pred_lstm = np.array([x[0] for x in data_lstm['pred']])

# Ограничим для наглядности
N = 300
x = np.arange(N)

# Построение графика
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# GRU
axs[0].plot(x, y_true_gru[:N], label='Real Values', color='blue')
axs[0].plot(x, y_pred_gru[:N], label='GRU Prediction', color='red')
axs[0].set_title('Prediction Result of GRU Model')
axs[0].legend()
axs[0].grid(True)

# LSTM
axs[1].plot(x, y_true_lstm[:N], label='Real Values', color='blue')
axs[1].plot(x, y_pred_lstm[:N], label='LSTM Prediction', color='red')
axs[1].set_title('Prediction Result of LSTM Model')
axs[1].legend()
axs[1].grid(True)

plt.suptitle('Step Ahead = 1: Comparison Between GRU and LSTM')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('stepahead1_comparison.png', dpi=300)
plt.show()
