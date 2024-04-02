import numpy as np
import metric_learn
import pandas as pd
import json
import decimal
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

X = []
Y = []

opponent = "OSU"

reactions = ["Booing", "Cheering", "Moving", "Postgame", "Storming", "Ugh"]
#reactions = ["Booing", "Cheering", "Storming", "Ugh"]

sensor_nodes = ['S-13','S-15','S-16','S-21','S-22','S-23','S-25']
#sensor_nodes = ['S-23']
for reaction in reactions:
    for node in sensor_nodes:
        plot_count = 0
        f = open("MICH_"+str(opponent)+"_file_labels.json")
        file_labels = json.load(f)
        for file in file_labels:
            if (file_labels[file] != reaction):
                continue
            plot_count += 1

            try:
                df = pd.read_csv("supervised_metric_learning/features_csvs/"+reaction+"/"+node+"_"+reaction+str(plot_count)+"_"+opponent+".csv")
            except (FileNotFoundError):
                break

            d = dict(df.to_numpy())

            RMS     = float(d['RMS'])
            VAR     = float(d['VAR'])
            STD     = float(d['STD'])
            PtoP    = float(d['P2P'])
            CREST   = float(d['CREST FACTOR'])
            SKEW    = float(d['SKEW'].replace('[','').replace(']',''))
            KURT    = float(d['KURTOSIS'].replace('[','').replace(']',''))

            X.append([RMS, VAR, PtoP, CREST])
            Y.append(reaction)

    print(reaction+" finished extracting")

print("Beginning fit")

model = metric_learn.LMNN(n_neighbors=5, max_iter=100)
start = time.time()
model.fit(X, Y)
end = time.time()
print("Took " + str(int(end - start)) + " sec")

X_model = model.transform(X)

metric_func = model.get_metric()

plt.figure(figsize=(8, 6))
colormap=plt.cm.Paired

# clean the figure
plt.clf()

colors = {'Booing': 'tomato',
          'Postgame': 'black',
          'Storming': 'orchid',
          'Ugh': 'blue',
          'Moving': 'yellow',
          'Cheering': 'green'}

Y_colors = []
for y in Y:
    Y_colors.append(colors[y])

tsne = TSNE()
X_embedded = tsne.fit_transform(X_model)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=np.asarray(Y_colors), cmap=colormap)

plt.xticks(())
plt.yticks(())

plt.show()
