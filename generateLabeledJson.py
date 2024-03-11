import json
import matplotlib.pyplot as plt
import glob
import os
import csv
# data path: ["GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/", "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-purdue/"]

csvpath = "EECS 598_Data-MichiganOSU.csv"

#prevEndTime = 0
#seg_data = []
#times = []

timesDict = {}
labelsDict = {}

for i, fp in enumerate(glob.glob("GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/*")[:]):
    numSamples = 0
    print(fp)
    f = open(fp)
    data = json.load(f)
    startTime = data[0]['timestamp']
    endTime = data[-1]['timestamp'] + len(data[-1]['data']) # account for last set of samples (1ms per sample)

    #print(startTime)
    #print(endTime)
    #print(startTime - prevEndTime)
    #prevEndTime = endTime
    #for d in data:
    #    seg_data += d['data']
    #    numSamples += len(d['data'])
    #times += [int(startTime + x*(endTime-startTime)/numSamples) for x in range(numSamples)]

    timesDict[fp] = [startTime, endTime]
    labelsDict[fp] = "Unlabeled"

csv_labels = []

# read csv file to a list of dictionaries
with open(csvpath, 'r') as file:
    csv_reader = csv.DictReader(file)
    csv_labels = [row for row in csv_reader]

for i in range(len(csv_labels)):
    for file in timesDict:

        if (i+1 == len(csv_labels) and timesDict[file][0] > int(csv_labels[i]['UNIX Timestamp (ms)'])): # prevent out of range, fix TODO (how to handle end of game files?)
            # Last file, way past end of game
            continue
        elif (timesDict[file][0] > int(csv_labels[i]['UNIX Timestamp (ms)'])):
            # 15 * 60 * 1000 = 15min in ms
            if (timesDict[file][1] > int(csv_labels[-1]['UNIX Timestamp (ms)']) + (15 * 60 * 1000)): 
                continue # discard files past end of game
            # pick label according to file midpoint
            if((timesDict[file][1] - (timesDict[file][0]) / 2) + timesDict[file][0] < int(csv_labels[i+1]['UNIX Timestamp (ms)'])):
                labelsDict[file] = csv_labels[i]['Reaction']
            else:
                labelsDict[file] = csv_labels[i+1]['Reaction']

# Serializing json
file_labels = json.dumps(labelsDict, indent=4)
file_times = json.dumps(timesDict, indent=4)
 
# Writing to file labels json
with open("MICH_OSU_file_labels.json", "w") as outfile:
    outfile.write(file_labels)

# Writing to file times json
with open("MICH_OSU_file_times.json", "w") as outfile:
    outfile.write(file_times)

#
#     # plt.title("S-15")
#     # plt.plot(seg_data, label=sensor_fp.split("_")[-1], alpha=0.5)
# points = list(zip(times, seg_data))
# plt.plot(points, alpha=0.5)

# plt.xticks()
# plt.ylim(0000, 5000)
# #plt.legend(loc="upper right")

# plt.savefig("plot.png")

