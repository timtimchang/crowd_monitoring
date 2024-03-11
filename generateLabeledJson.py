import json
import matplotlib.pyplot as plt
import glob
import os
import csv

class GenJSON:

    data_paths = {'OSU': "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/", 'Purdue': "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-purdue/"}
    csv_paths = {'OSU': "EECS 598_Data-MichiganOSU.csv", 'Purdue': "EECS 598_Data-MichiganOSU.csv"}

    opponent = ''
    timesDict = {}
    labelsDict = {}

    def run(self):
        while (self.opponent not in self.data_paths):
            print("Select valid opponent to generate JSONs: ")
            for opp in self.data_paths:
                print("    " + opp)
            self.opponent = input()

        # TEMP
        if (self.opponent == 'Purdue'):
            print('Purdue data not yet available')
            return
        
        self.populateTimes()
        self.populateLabels()
        self.generateJSONs()

    def populateTimes(self):

        for i, fp in enumerate(glob.glob(self.data_paths[self.opponent]+ "/*")[:]):
            f = open(fp)
            data = json.load(f)
            startTime = data[0]['timestamp']
            endTime = data[-1]['timestamp'] + len(data[-1]['data']) # account for last set of samples (1ms per sample)

            self.timesDict[fp] = [startTime, endTime]
            self.labelsDict[fp] = "Unlabeled"

    def populateLabels(self):
        csv_labels = []

        # read csv file to a list of dictionaries
        with open(self.csv_paths[self.opponent], 'r') as file:
            csv_reader = csv.DictReader(file)
            csv_labels = [row for row in csv_reader]

        for i in range(len(csv_labels)):
            for file in self.timesDict:

                if (i+1 == len(csv_labels) and self.timesDict[file][0] > int(csv_labels[i]['UNIX Timestamp (ms)'])): # prevent out of range, fix TODO (how to handle end of game files?)
                    # Last file, way past end of game
                    continue
                elif (self.timesDict[file][0] > int(csv_labels[i]['UNIX Timestamp (ms)'])):
                    # 15 * 60 * 1000 = 15min in ms
                    if (self.timesDict[file][1] > int(csv_labels[-1]['UNIX Timestamp (ms)']) + (15 * 60 * 1000)): 
                        continue # discard files past end of game
                    # pick label according to file midpoint
                    if((self.timesDict[file][1] - (self.timesDict[file][0]) / 2) + self.timesDict[file][0] < int(csv_labels[i+1]['UNIX Timestamp (ms)'])):
                        self.labelsDict[file] = csv_labels[i]['Reaction']
                    else:
                        self.labelsDict[file] = csv_labels[i+1]['Reaction']

    def generateJSONs(self):

        # Serializing json
        file_labels = json.dumps(self.labelsDict, indent=4)
        file_times = json.dumps(self.timesDict, indent=4)
        
        # Writing to file labels json
        with open("MICH_"+str(self.opponent)+"_file_labels.json", "w") as outfile:
            outfile.write(file_labels)

        # Writing to file times json
        with open("MICH_"+str(self.opponent)+"_file_times.json",'Purdue' "w") as outfile:
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
    
def main():
    global data_paths

    generator = GenJSON()
    generator.run()

if __name__ == "__main__":
    main()

