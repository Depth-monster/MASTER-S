import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Путь к корню проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Пути к GRU и LSTM предсказаниям
gru_path = os.path.join(ROOT_DIR, 'results', 'gru', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')
lstm_path = os.path.join(ROOT_DIR, 'results', 'lstm', 'timestep_60', 'lb=60_la=20_ne1=16_ne2=16_ne=16', 'values.pickle')

# Загрузка данных
with open(gru_path, 'rb') as f:
    data_gru = pickle.load(f)

with open(lstm_path, 'rb') as f:
    data_lstm = pickle.load(f)

# Извлечение значений
#y_true = np.array(data_gru['real'])        # одинаковые
#y_pred_gru = np.array(data_gru['pred'])
#y_pred_lstm = np.array(data_lstm['pred'])
import itertools

# Плоские массивы
y_true = np.array(list(itertools.chain.from_iterable(data_gru['real'])))
y_pred_gru = np.array(list(itertools.chain.from_iterable(data_gru['pred'])))
y_pred_lstm = np.array(list(itertools.chain.from_iterable(data_lstm['pred'])))



# Ограничим количество точек
N = 300
y_true = y_true[:N]
y_pred_gru = y_pred_gru[:N]
y_pred_lstm = y_pred_lstm[:N]

# Построение графиков
plt.figure(figsize=(12, 5))

# GRU
plt.subplot(1, 2, 1)
plt.plot(y_true, label='Real Values', color='blue')
plt.plot(y_pred_gru, label='GRU Prediction', color='red')
plt.title('Prediction Result of GRU Model')
plt.legend()
plt.grid(True)

# LSTM
plt.subplot(1, 2, 2)
plt.plot(y_true, label='Real Values', color='blue')
plt.plot(y_pred_lstm, label='LSTM Prediction', color='red')
plt.title('Prediction Result of LSTM Model')
plt.legend()
plt.grid(True)

# Общий заголовок
plt.suptitle('Difference of the Test Result Between Two RNN Variants')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Сохранение
plt.savefig('gru_vs_lstm_comparison.png', dpi=300)
plt.show()
