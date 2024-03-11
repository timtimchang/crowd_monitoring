import json
import matplotlib.pyplot as plt
import glob
import os
# data path: ["GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/", "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-purdue/"]

prevEndTime = 0

for i, sensor_fp in enumerate(glob.glob("GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/*")[:5]):
    seg_data = []
    print(sensor_fp)
    f = open(sensor_fp)
    data = json.load(f)
    startTime = data[0]['timestamp']
    endTime = data[-1]['timestamp']
    print(startTime)
    print(endTime)
    print(startTime - prevEndTime)
    prevEndTime = endTime

    # plt.title("S-15")
    # plt.plot(seg_data, label=sensor_fp.split("_")[-1], alpha=0.5)
    plt.plot(seg_data, label=f"Sensor-23", alpha=0.5)

plt.xticks([])
plt.ylim(0000, 5000)
plt.legend(loc="upper right")