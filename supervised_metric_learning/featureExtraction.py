import requests, zipfile 
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.fft import fft, fftfreq

import json

FEATURES = ['MIN','MAX','MEAN','RMS','VAR','STD','POWER','PEAK','P2P','CREST FACTOR','SKEW','KURTOSIS',
            'MAX_f','SUM_f','MEAN_f','VAR_f','PEAK_f','SKEW_f','KURTOSIS_f']

def features_extraction(df): 
    
    Min=[];Max=[];Mean=[];Rms=[];Var=[];Std=[];Power=[];Peak=[];Skew=[];Kurtosis=[];P2p=[];CrestFactor=[];
    FormFactor=[]; PulseIndicator=[];
    Max_f=[];Sum_f=[];Mean_f=[];Var_f=[];Peak_f=[];Skew_f=[];Kurtosis_f=[]
    
    X = df.values
    ## TIME DOMAIN ##

    Min.append(np.min(X))
    Max.append(np.max(X))
    Mean.append(np.mean(X))
    Rms.append(np.sqrt(np.mean(X**2)))
    Var.append(np.var(X))
    Std.append(np.std(X))
    Power.append(np.mean(X**2))
    Peak.append(np.max(np.abs(X)))
    P2p.append(np.ptp(X))
    CrestFactor.append(np.max(np.abs(X))/np.sqrt(np.mean(X**2)))
    Skew.append(stats.skew(X))
    Kurtosis.append(stats.kurtosis(X))
    FormFactor.append(np.sqrt(np.mean(X**2))/np.mean(X))
    PulseIndicator.append(np.max(np.abs(X))/np.mean(X))
    ## FREQ DOMAIN ##
    ft = fft(X)
    S = np.abs(ft**2)/len(df)
    Max_f.append(np.max(S))
    Sum_f.append(np.sum(S))
    Mean_f.append(np.mean(S))
    Var_f.append(np.var(S))
    
    Peak_f.append(np.max(np.abs(S)))
    Skew_f.append(stats.skew(X))
    Kurtosis_f.append(stats.kurtosis(X))
    #Create dataframe from features
    df_features = pd.DataFrame(index = [FEATURES], 
                               data = [Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis,
                                       Max_f,Sum_f,Mean_f,Var_f,Peak_f,Skew_f,Kurtosis_f])
    return df_features


sensor_nodes = ['S-13','S-15','S-16','S-21','S-22','S-23','S-25']

reactions = ["Unlabeled","Booing","Cheering","Postgame","Storming","Ugh"]
opponent = "OSU"

for node in sensor_nodes:
    for reaction in reactions:
        f = open(node+"_MICH_"+str(opponent)+"_file_labels.json")
        file_labels = json.load(f)
        plot_count = 0
        for file in file_labels:
            if (file_labels[file] != reaction):
                continue
            plot_count += 1

            f2 = open(file)
            info = json.load(f2)
            data = []
            for d in info:
                data += (d['data'])
            df = pd.DataFrame({'data': data})
            mean = df['data'].mean()
            df['data'] = df['data'] - mean

            features = features_extraction(df)

            try:
                features.to_csv("supervised_metric_learning/features_csvs/" + str(reaction) + "/" + node+"_"+str(reaction) + str(plot_count) + "_" + str(opponent) +  ".csv")
            except(OSError):
                # Create directory if it doesn't exist yet
                os.makedirs("supervised_metric_learning/features_csvs/" + str(reaction) + "/" )
                features.to_csv("supervised_metric_learning/features_csvs/" + str(reaction) + "/" + node+"_"+ str(reaction) + str(plot_count) + "_" + str(opponent) +  ".csv")
    