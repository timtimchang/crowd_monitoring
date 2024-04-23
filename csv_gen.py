import json
from generateLabeledJson import GenJSON
import matplotlib.pyplot as plt
import glob
import os
from scipy.fft import fft, fftfreq
import pandas as pd
import numpy as np

all_data = []
times = []
opponent = 'Purdue'

first_ts = 0
first_ts_set = False

f = open("S-23_MICH_"+str(opponent)+"_file_labels.json")
file_labels = json.load(f)
plot_count = 0
for file in file_labels:
    plot_count += 1
    
    f2 = open(file)
    info = json.load(f2)
    if not first_ts_set:
        first_ts = info[0]['timestamp']
        first_ts_set=True
    for d in info:
        for i in range(0, len(d['data'])):
            all_data.append(d['data'][i] / info[0]['gain'])  # Adjusted to divide by the gain from the first element
            times.append(d['timestamp'] + i - first_ts)


df = pd.DataFrame({'data': all_data, 'times': times})
mean = df['data'].mean()
df['data'] = df['data'] - mean

df.to_csv("Purdue_S-23_full_game.csv")