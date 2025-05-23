import pickle
import numpy as np

with open('results/gru/timestep_60/lb=60_la=20_ne1=16_ne2=16_ne=16/values.pickle', 'rb') as f:
    data = pickle.load(f)

print("type(data['real']):", type(data['real']))
print("len(data['real']):", len(data['real']))
print("shape of first entry:", np.array(data['real'][0]).shape)
