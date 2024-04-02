import json
import glob
import csv

class GenJSON:
    sensor_nodes = ['S-13','S-15','S-16','S-21','S-22','S-23','S-25']
    data_paths = {'OSU': "GEOSCOPE_SENSOR_S-xx/GEOSCOPE_SENSOR_S-xx-ohio/", 'Purdue': "GEOSCOPE_SENSOR_S-xx/GEOSCOPE_SENSOR_S-xx-purdue/"}
    csv_paths = {'OSU': "EECS 598_Data-MichiganOSU.csv", 'Purdue': "EECS 598_Data-MichiganOSU.csv"}

    opponent = ''
    timesDict = {}
    labelsDict = {}

    node = ''

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
        
        for node in self.sensor_nodes:
            self.node = node
            self.timesDict = {}
            self.labelsDict = {}

            self.populateTimes()
            self.populateLabels()
            self.generateJSONs()
    
    def run_noinput(self, opp: str):
        self.opponent = opp
        if (self.opponent not in self.data_paths):
            raise ValueError("Invalid opponent")

        # TEMP
        if (self.opponent == 'Purdue'):
            raise ValueError('Purdue data not yet available')
        
        self.populateTimes()
        self.populateLabels()
        self.generateJSONs()

    def populateTimes(self):

        path = self.data_paths[self.opponent]+ "/*"
        print(path)
        path = path.replace("S-xx", self.node)
        print(self.node)
        print(path)

        for i, fp in enumerate(glob.glob(path)[:]):
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

        for file in self.timesDict:
            for i in range(len(csv_labels)):

                # if file timestamp midpoint is within a game time range it assigns that label to the file. TODO Improve this conditional
                timeMidpoint = ((self.timesDict[file][1] - self.timesDict[file][0]) / 2) + self.timesDict[file][0]

                if (timeMidpoint > int(csv_labels[i]['UNIX Timestamp (ms)'])): 
                    if (timeMidpoint < int(csv_labels[i]['UNIX Timestamp (ms)']) + self.minutes_to_ms(1)): 
                        # Assign label if less than 1 minute past end of current play event and less than 15 minutes past end of game

                        self.labelsDict[file] = csv_labels[i]['Reaction']
                    elif (timeMidpoint > (int(csv_labels[-1]['UNIX Timestamp (ms)'])) and timeMidpoint < (int(csv_labels[-1]['UNIX Timestamp (ms)']) + self.minutes_to_ms(15))):
                        # Post game section
                        self.labelsDict[file] = csv_labels[i]['Reaction'] # storming
                    elif (timeMidpoint > (int(csv_labels[-1]['UNIX Timestamp (ms)']) + self.minutes_to_ms(15))):
                        self.labelsDict[file] = "Postgame"
                    #else leave unlabeled (media timeout, etc.)

    def generateJSONs(self):

        # Serializing json
        file_labels = json.dumps(self.labelsDict, indent=4)
        file_times = json.dumps(self.timesDict, indent=4)
        
        # Writing to file labels json
        with open(self.node+"_MICH_"+str(self.opponent)+"_file_labels.json", "w") as outfile:
            outfile.write(file_labels)

        # Writing to file times json
        with open(self.node+"_MICH_"+str(self.opponent)+"_file_times.json","w") as outfile:
            outfile.write(file_times)

    def minutes_to_ms(self, min: int) -> int:
        return min * 60 * 1000
    
def main():
    generator = GenJSON()
    generator.run()

if __name__ == "__main__":
    main()
