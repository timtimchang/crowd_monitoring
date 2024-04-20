import requests, zipfile 
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.fft import fft, fftfreq
from scipy.integrate import cumtrapz
from scipy import signal

import json

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

FEATURES = ['MIN','MAX','MEAN','RMS','VAR','STD','POWER','PEAK','P2P','CREST FACTOR','SKEW','KURTOSIS',
            'MAX_f','SUM_f','MEAN_f','VAR_f','PEAK_f','SKEW_f','KURTOSIS_f','Q1_Pwr_f','Q2_Pwr_f','Q3_Pwr_f',
            'High_F_Pwr_f','Low_F_Pwr_f','Med1_F_Pwr_f','Med2_F_Pwr_f','Highest_Peak_f']

def features_extraction(df): 

    sample_rate = 1000
    
    Min=[];Max=[];Mean=[];Rms=[];Var=[];Std=[];Power=[];Peak=[];Skew=[];Kurtosis=[];P2p=[];CrestFactor=[]
    FormFactor=[]; PulseIndicator=[]
    Max_f=[];Sum_f=[];Mean_f=[];Var_f=[];Peak_f=[];Skew_f=[];Kurtosis_f=[]

    Q1_Pwr_f=[];Q2_Pwr_f=[];Q3_Pwr_f=[]
    High_F_Pwr_f=[]
    Low_F_Pwr_f=[]
    Med1_F_Pwr_f=[]
    Med2_F_Pwr_f=[]
    Highest_Peak_f=[]

    X = df['data'].values
    
    _, X_lmax = hl_envelopes_idx(X, dmax=50)
    X_env = X[X_lmax]

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

    # freqs = abs(np.fft.fftfreq(len(df['data'])) * sample_rate)[:len(ft)//2]
    # ft_oneside = np.asarray(abs(ft)).squeeze()[:len(ft)//2]

    ### PSD Statistics
    f, Pxx_den = signal.periodogram(X, 1000)
    _, lmax = hl_envelopes_idx(Pxx_den, dmax=50)

    f_env = f[lmax]
    P_env = Pxx_den[lmax]

    integral = cumtrapz(P_env, f_env, initial=0)
    if integral.size == 0:
        return pd.DataFrame()
    Q4_value = integral[-1]

    Q1_index = 0
    Q2_index = 0
    Q3_index = 0

    for val in integral:
        if val < (Q4_value / 4):
            Q1_index += 1 # val < 1/4 total power
        if val < (Q4_value / 2):
            Q2_index += 1 # val < 1/2 total power
        if val < (Q4_value * 3/4):
            Q3_index += 1 # val < 3/4 total power

    Q1_Pwr_f.append(f_env[Q1_index])
    Q2_Pwr_f.append(f_env[Q2_index])
    Q3_Pwr_f.append(f_env[Q3_index])

    # integral_hi = cumtrapz(ft_oneside[Q3_index:], freqs[Q3_index:], initial=0)
    # integral_lo = cumtrapz(ft_oneside[0:Q1_index], freqs[0:Q1_index], initial=0)
    # integral_med = cumtrapz(ft_oneside[Q1_index:Q2_index], freqs[Q1_index:Q2_index], initial=0)
    High_F_Pwr_f.append(np.sqrt(np.mean(P_env[Q3_index:]**2)))
    Med1_F_Pwr_f.append(np.sqrt(np.mean(P_env[Q1_index:Q2_index]**2)))
    Med2_F_Pwr_f.append(np.sqrt(np.mean(P_env[Q2_index:Q3_index]**2)))
    Low_F_Pwr_f.append(np.sqrt(np.mean(P_env[:Q1_index]**2)))

    Highest_Peak_f.append(f_env[np.argmax(np.abs(P_env))])

    #Create dataframe from features
    df_features = pd.DataFrame(index = [FEATURES], 
                               data = [Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis,
                                       Max_f,Sum_f,Mean_f,Var_f,Peak_f,Skew_f,Kurtosis_f,Q1_Pwr_f,Q2_Pwr_f,Q3_Pwr_f,
                                       High_F_Pwr_f,Low_F_Pwr_f,Med1_F_Pwr_f,Med2_F_Pwr_f,Highest_Peak_f])
    return df_features


#sensor_nodes = ['S-13','S-15','S-16','S-21','S-22','S-23','S-25']
sensor_nodes = ['S-23']

reactions = ["Unlabeled","Booing","Cheering","Postgame","Storming","Ugh","Moving"]
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
                data += [elt / (info[0]['gain']) for elt in d['data']]
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
    
print("Completed feature extraction")