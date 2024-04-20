import json
from generateLabeledJson import GenJSON
import matplotlib.pyplot as plt
import glob
import os
from scipy.fft import fft, fftfreq
import pandas as pd
import numpy as np
import math
from scipy import signal
import datetime

from scipy.signal import butter, lfilter
from scipy.signal import freqs

def butter_lowpass(cutOff, fs, order=5):
    return butter(order, cutOff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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


class PlottingAssistant:

    data_paths = {'OSU': "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/", 'Purdue': "GEOSCOPE_SENSOR_S-22/GEOSCOPE_SENSOR_S-22-purdue/"}

    sample_rate = 1000

    cutOff = 1500 #cutoff frequency in rad/s
    fs = 6200 #sampling frequency in rad/s
    order = 10 #order of filter

    def reactionPlots(self, opponent: str, reaction: str,):
        # Remove old plots in specified folder
        # for filename in glob.glob('Plots/'+reaction+'/*.png'):
        #     os.remove(filename)

        f = open("MICH_"+str(opponent)+"_file_labels.json")
        file_labels = json.load(f)
        plot_count = 0
        for file in file_labels:
            if (file_labels[file] != reaction):
                continue
            plot_count += 1
            plt.close()
            plt.title("S-23: " + reaction + " #" + str(plot_count) + " (" + opponent + ")")
            
            f2 = open(file)
            info = json.load(f2)
            data = []
            for d in info:
                data += [elt / (info[0]['gain']) for elt in d['data']]
            
            df = pd.DataFrame({'data': data})
            mean = df['data'].mean()
            df['data'] = df['data'] - mean
            plt.plot(df['data'], alpha=0.5)

            _, X_lmax = hl_envelopes_idx(df['data'], dmax=20)
            X_env = df['data'][X_lmax]
            
            plt.plot(X_env, alpha=0.5, color='red')

            plt.xticks()
            # plt.ylim(0000, 5000)
            #plt.legend(loc="upper right")

            plt.show()

            # try:
            #     plt.savefig("Plots/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")
            # except(FileNotFoundError):
            #     # Create directory if it doesn't exist yet
            #     os.makedirs("Plots/"+reaction+"/", exist_ok=True)
            #     plt.savefig("Plots/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")

    def plot_all(self, opponent: str):

        all_data = []
        times = []

        first_ts = 0
        first_ts_set = False

        # Remove old plots in specified folder
        f = open("MICH_"+str(opponent)+"_file_labels.json")
        file_labels = json.load(f)
        plot_count = 0
        for file in file_labels:
            plot_count += 1
            plt.close()
            plt.title("S-23: #" + str(plot_count) + " (" + opponent + ")")
            
            f2 = open(file)
            info = json.load(f2)
            if not first_ts_set:
                first_ts = info[0]['timestamp']
                first_ts_set=True
            for d in info:
                for i in range(0, len(d['data'])):
                    all_data.append(d['data'][i] / info[0]['gain'])  # Adjusted to divide by the gain from the first element
                    times.append(d['timestamp'] + i - first_ts)
        
        df = pd.DataFrame({'data': all_data})

        mean = df['data'].mean()
        df['data'] = df['data'] - mean
        plt.plot(times, df['data'], alpha=0.5)        

        _, X_lmax = hl_envelopes_idx(df['data'], dmax=20)
        X_env = df['data'][X_lmax]
        t_env = [times[index] for index in X_lmax]
        
        plt.plot(t_env, X_env, alpha=0.5, color='red')

        plt.xticks()
        # plt.ylim(0000, 5000)
        #plt.legend(loc="upper right")

        plt.show()
        

    def plot_f_domain(self, opponent: str, reaction: str,):

        # Remove old plots in specified folder
        # for filename in glob.glob('Plots/FDomain/'+reaction+'/*.png'):
        #     os.remove(filename)

        f = open("S-23_MICH_"+str(opponent)+"_file_labels.json")
        file_labels = json.load(f)
        plot_count = 0
        for file in file_labels:
            if (file_labels[file] != reaction):
                continue
            plot_count += 1
            plt.close()
            plt.title("S-23, Freq. Domain: " + reaction + " #" + str(plot_count) + " (" + opponent + ")")
            
            f2 = open(file)
            info = json.load(f2)
            data = []
            for d in info:
                data += [elt / (info[0]['gain']) for elt in d['data']]
            
            df = pd.DataFrame({'data': data})
            mean = df['data'].mean()
            df['data'] = df['data'] - mean

            ft = fft(np.asarray(df['data']))
            freqs = abs(np.fft.fftfreq(len(df['data'])) * self.sample_rate) # Hz

            plt.yscale('log', base=10)

            plt.plot(freqs[1:len(ft)//2], (abs(ft))[1:len(ft)//2], alpha=0.5)

            plt.xticks()
            plt.ylim(1e1, 1e7)
            #plt.legend(loc="upper right")

            try:
                plt.savefig("Plots/FDomain/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")
            except(FileNotFoundError):
                # Create directory if it doesn't exist yet
                os.makedirs("Plots/FDomain/"+reaction+"/", exist_ok=True)
                plt.savefig("Plots/FDomain/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")

    def plot_spectrogram(self, opponent: str, reaction: str,):

        # Remove old plots in specified folder
        # for filename in glob.glob('Plots/FDomain/'+reaction+'/*.png'):
        #     os.remove(filename)

        f = open("S-23_MICH_"+str(opponent)+"_file_labels.json")
        file_labels = json.load(f)
        plot_count = 0
        for file in file_labels:
            if (file_labels[file] != reaction):
                continue
            plot_count += 1
            plt.close()
            plt.title("S-23, Spectrogram: " + reaction + " #" + str(plot_count) + " (" + opponent + ")")
            
            f2 = open(file)
            info = json.load(f2)
            data = []
            for d in info:
                data += [elt / (info[0]['gain']) for elt in d['data']]
            
            df = pd.DataFrame({'data': data})
            mean = df['data'].mean()
            df['data'] = df['data'] - mean
            
            # Matplotlib.pyplot.specgram() function to
            # generate spectrogram
            plt.specgram(df['data'], Fs=2, cmap="rainbow")

            plt.xticks()
            #plt.legend(loc="upper right")

            plt.show()

            # try:
            #     plt.savefig("Plots/Spectrogram/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")
            # except(FileNotFoundError):
            #     # Create directory if it doesn't exist yet
            #     os.makedirs("Plots/Spectrogram/"+reaction+"/", exist_ok=True)
            #     plt.savefig("Plots/Spectrogram/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")

    def plot_PSD(self, opponent: str, reaction: str,):

        # Remove old plots in specified folder
        # for filename in glob.glob('Plots/FDomain/'+reaction+'/*.png'):
        #     os.remove(filename)

        f = open("S-23_MICH_"+str(opponent)+"_file_labels.json")
        file_labels = json.load(f)
        plot_count = 0
        for file in file_labels:
            if (file_labels[file] != reaction):
                continue
            plot_count += 1
            plt.close()
            plt.title("S-23, Spectrogram: " + reaction + " #" + str(plot_count) + " (" + opponent + ")")
            
            f2 = open(file)
            info = json.load(f2)
            data = []
            for d in info:
                data += [elt / (info[0]['gain']) for elt in d['data']]
            
            df = pd.DataFrame({'data': data})
            mean = df['data'].mean()
            df['data'] = df['data'] - mean

            f, Pxx_den = signal.periodogram(df['data'], 1000)
            plt.semilogy(f[1:], Pxx_den[1:])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD')

            _, lmax = hl_envelopes_idx(Pxx_den, dmax=50)
            plt.semilogy(f[lmax], Pxx_den[lmax],color='red')

            plt.xticks()
            #plt.legend(loc="upper right")

            plt.show()

            # try:
            #     plt.savefig("Plots/Spectrogram/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")
            # except(FileNotFoundError):
            #     # Create directory if it doesn't exist yet
            #     os.makedirs("Plots/Spectrogram/"+reaction+"/", exist_ok=True)
            #     plt.savefig("Plots/Spectrogram/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")


def main():
    opponent = 'OSU'

    # Ensure labels are up-to-date
    # print("Updating relevant JSON files")
    # generator = GenJSON()
    # generator.run_noinput(opponent)
    # print("Complete")


    print("Generating specified visualizations")
    plotter = PlottingAssistant()
    plotter.reactionPlots(opponent, "Cheering")
    # plotter.reactionPlots(opponent, "Booing")
    # plotter.reactionPlots(opponent, "Moving")
    # plotter.reactionPlots(opponent, "Storming")
    # plotter.reactionPlots(opponent, "Ugh")
    # plotter.reactionPlots(opponent, "Postgame")
    
    # plotter.plot_f_domain(opponent, "Cheering")
    # plotter.plot_f_domain(opponent, "Booing")
    # plotter.plot_f_domain(opponent, "Moving")
    # plotter.plot_f_domain(opponent, "Storming")
    # plotter.plot_f_domain(opponent, "Ugh")
    # plotter.plot_f_domain(opponent, "Postgame")
    # plotter.plot_f_domain(opponent, "Unlabeled")

    # plotter.plot_PSD(opponent, "Postgame") 

    print("Complete")

if __name__ == "__main__":
    main()
