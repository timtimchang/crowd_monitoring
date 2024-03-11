import json
import matplotlib.pyplot as plt
import glob
import os
# data path: ["GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/", "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-purdue/"]

prevEndTime = 0
seg_data = []
times = []

for i, sensor_fp in enumerate(glob.glob("GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/*")[:10]):
    numSamples = 0
    print(sensor_fp)
    f = open(sensor_fp)
    data = json.load(f)
    startTime = data[0]['timestamp']
    endTime = data[-1]['timestamp']
    print(startTime)
    print(endTime)
    print(startTime - prevEndTime)
    prevEndTime = endTime
    for d in data:
        seg_data += d['data']
        numSamples += len(d['data'])
    endTime += len(data[-1]['data']) # account for last set of samples
    times += [startTime + x*(endTime-startTime)/numSamples for x in range(numSamples)]


    # plt.title("S-15")
    # plt.plot(seg_data, label=sensor_fp.split("_")[-1], alpha=0.5)
    points = list(zip(times, seg_data))
    plt.plot(points, label=f"Sensor-23: "+str(startTime), alpha=0.5)

plt.xticks([])
plt.ylim(0000, 5000)
plt.legend(loc="upper right")

plt.savefig("plot.png")
