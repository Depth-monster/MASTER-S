import pickle

with open('results/gru/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle', 'rb') as f:
    data = pickle.load(f)

print("Ключи в pickle-файле:", data.keys())
