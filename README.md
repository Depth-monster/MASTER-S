Проект представляет из себя модуль прогнозирования сетевого трафика. Трафик был взят из открытх источников и отформатирован для проведения проектной работы.
Приведу шаги, которые необходимы, чтобы модуль работал корректно.
Что понадобится для запуска:
Ubuntu сервер
Python 3.7.4
и все необходимые библиотеки. Команды по шагам ниже:

sudo apt update

sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
sudo tar xzf Python-3.7.4.tgz
cd Python-3.7.4
sudo ./configure --enable-optimizations
sudo make -j$(nproc)
sudo make altinstall
![image](https://github.com/user-attachments/assets/cd8be04c-ce90-4538-8867-4404a84f0f32)

После того, как закончатя тесты, можно приступить к клонированию репозитория.

git clone https://github.com/Depth-monster/Network-Traffic-Prediction.git
cd Network-Traffic-Prediction
python3.7 -m venv venv
source venv/bin/activate
sudo nano requirements.txt
pip install -r requirements.txt
sudo apt update
sudo apt install -y libhdf5-dev
pip install h5py
pip install -r requirements.txt
Команды лучше выполнять в коренном репо Network-....
Обучение модели происзодит так:

python3 main/gru.py 60 20 16 16 16 60

На вход подается:
60 - look_back - Сколько предыдущих точек времени (в минутах) используется в качестве входа модели (Step Back = 60 минут) 
20 - look_ahead - Сколько точек времени вперёд модель должна предсказать (Step Ahead = 20 минут).
16 - neuron1 (encoder) - Кол-во нейронов в первом слое модели (энкодере).
16 - neuron2 (decoder1) - Кол-во нейронов в первом декодере (decoder layer 1).
16 - neuron3 (decoder2) - Кол-во нейронов в втором декодере (decoder layer 2).
60 - time_step - Интервал между временными шагами — обычно совпадает с look_back и нужен для формирования подпапки timestep_60/ в структуре вывода.

Что делает gru.py при запуске:
Загружает данные (data/network_traffic.csv).
Формирует обучающую выборку на основе последних 60 минут.
Обучает GRU-модель предсказывать 20 следующих точек.
Архитектура модели:
1 GRU-слой (16 нейронов)
2 дополнительных GRU-декодера по 16 нейронов
выход: TimeDistributed(Dense(1)) для предсказания временного ряда.
Результаты:
сохраняются в results/gru/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/
values.pickle — предсказания
metrics_comparison.csv — MAPE по шагам
Аналогично lstm.py
![image](https://github.com/user-attachments/assets/e4e60d9f-5be8-45f5-b158-9b876849d50c)

![gru_lstm_vs_real_separate](https://github.com/user-attachments/assets/48332bfd-dcf2-47c7-a813-3ad41f596fe6)
![mape_comparison_gru_lstm](https://github.com/user-attachments/assets/e683b6bb-c062-4053-936d-da8cec6dad7d)
![mape_summary_table](https://github.com/user-attachments/assets/30f54cde-7151-4f62-bc92-959f20dec2db)


