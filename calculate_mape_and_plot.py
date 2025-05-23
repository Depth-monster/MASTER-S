import pandas as pd
import matplotlib.pyplot as plt
import os

# Пути к CSV-файлам с метриками
gru_path = 'results/gru/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/metrics_comparison.csv'
lstm_path = 'results/lstm/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/metrics_comparison.csv'

# Загрузка CSV
gru_df = pd.read_csv(gru_path)
lstm_df = pd.read_csv(lstm_path)

# Извлечение только MAPE-колонок
gru_mape = gru_df.filter(like='MAPE')
lstm_mape = lstm_df.filter(like='MAPE')

# Расчёт среднего MAPE
gru_avg = gru_mape.values.flatten().mean()
lstm_avg = lstm_mape.values.flatten().mean()

# Сохранение в таблицу
summary_df = pd.DataFrame({
    'Метрика': ['MAPE (средняя абсолютная процентная ошибка, %)'],
    'GRU': [round(gru_avg, 2)],
    'LSTM': [round(lstm_avg, 2)],
})

# Вывод в консоль
print(summary_df)

# Сохраняем как PNG-таблицу
fig, ax = plt.subplots(figsize=(6, 1.2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=summary_df.values,
                 colLabels=summary_df.columns,
                 cellLoc='center',
                 loc='center')

output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'mape_summary_table.png'), dpi=300)

# Сохраняем значения как CSV
summary_df.to_csv(os.path.join(output_dir, 'mape_summary_table.csv'), index=False)

print("✅ Готово! Таблица сохранена в output_plots/mape_summary_table.png и .csv")
