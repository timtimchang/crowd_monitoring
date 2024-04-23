import json
from generateLabeledJson import GenJSON
import matplotlib.pyplot as plt
import glob
import os
from scipy.fft import fft, fftfreq
import pandas as pd
import numpy as np
from supervised_metric_learning.featureExtraction import features_extraction

opponent = 'Purdue'
all_data = pd.read_csv(opponent+"_S-23_full_game.csv")['data']
times = pd.read_csv(opponent+"_S-23_full_game.csv")['times']

first_ts = 0
first_ts_set = False

curr_data = []
curr_times = []

start_time = times[0]

count = 1
for i in range(0,len(times)):
    # if in different 30s period
    if times[i] > start_time + (30 * 1000):
        start_time = times[i]
        df = pd.DataFrame({'data':curr_data, 'times': curr_times})
        # df.to_csv("OSU_half_minute_sections/section" + str(count) + ".csv")
        features = features_extraction(df)

        try:
            features.to_csv("supervised_metric_learning/features_csvs/"+opponent+"/section"+ str(count) +".csv")
        except(OSError):
            # Create directory if it doesn't exist yet
            os.makedirs("supervised_metric_learning/features_csvs/"+opponent+"/")
            features.to_csv("supervised_metric_learning/features_csvs/"+opponent+"/section"+ str(count) +".csv")

        count += 1
        curr_data = []
        curr_times = []

    
    curr_data.append(all_data[i])
    curr_times.append(times[i])