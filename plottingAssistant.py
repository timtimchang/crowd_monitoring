import json
from generateLabeledJson import GenJSON
import matplotlib.pyplot as plt
import glob
import os

class PlottingAssistant:

    data_paths = {'OSU': "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-ohio/", 'Purdue': "GEOSCOPE_SENSOR_S-23/GEOSCOPE_SENSOR_S-23-purdue/"}

    def reactionPlots(self, opponent: str, reaction: str,):
        # Remove old plots in specified folder
        for filename in glob.glob('Plots/'+reaction+'/*.png'):
            os.remove(filename)

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
                data += (d['data'])
            plt.plot(data, alpha=0.5)
            points = list(data)
            plt.plot(points, alpha=0.5)

            plt.xticks()
            plt.ylim(0000, 5000)
            #plt.legend(loc="upper right")

            try:
                plt.savefig("Plots/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")
            except(FileNotFoundError):
                # Create directory if it doesn't exist yet
                os.makedirs("Plots/"+reaction+"/", exist_ok=True)
                plt.savefig("Plots/"+reaction+"/"+reaction + str(plot_count) + "_" + opponent + ".png")

def main():
    opponent = 'OSU'

    # Ensure labels are up-to-date
    print("Updating relevant JSON files")
    generator = GenJSON()
    generator.run_noinput(opponent)
    print("Complete")

    print("Generating specified visualizations")
    plotter = PlottingAssistant()
    plotter.reactionPlots(opponent, "Cheering")
    plotter.reactionPlots(opponent, "Booing")
    plotter.reactionPlots(opponent, "Moving")
    plotter.reactionPlots(opponent, "Storming")
    plotter.reactionPlots(opponent, "Ugh")
    plotter.reactionPlots(opponent, "Postgame")
    print("Complete")

if __name__ == "__main__":
    main()
