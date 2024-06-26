import numpy as np
import metric_learn
import pandas as pd
import json
import decimal
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math

def plot_tsne(X, y, colormap=plt.cm.Paired):
    plt.figure(figsize=(8, 6))
    
    colors = {'Booing': 'tomato',
            'Postgame': 'black',
            'Storming': 'orchid',
            'Ugh': 'blue',
            'Moving': 'yellow',
            'Cheering': 'green',
            'Unlabeled': 'gray'}

    Y_colors = []
    for val in y:
        Y_colors.append(colors[val])

    # clean the figure
    plt.clf()

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_colors, cmap=colormap)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

    plt.xticks(())
    plt.yticks(())

    plt.show()

model = metric_learn.Covariance()

opponents = ['OSU']
# opponents = ['OSU', 'Purdue']
Train_X = {'OSU': [], 'Purdue': []}
Test_X = {'OSU': [], 'Purdue': []}
Train_Y = {'OSU': [], 'Purdue': []}
Test_Y = {'OSU': [], 'Purdue': []}
opp_X = {'OSU': [], 'Purdue': []}
opp_Y = {'OSU': [], 'Purdue': []}

reactions = ["Booing", "Cheering", "Moving", "Postgame", "Storming", "Ugh", "Unlabeled"]

count = 0

sensor_nodes = ['S-23']
for opponent in opponents:
    for node in sensor_nodes:
        X = []
        Y = []
        I = []
        # for reaction in reactions:
        #     plot_count = 0
        #     f = open(node + "_MICH_"+str(opponent)+"_file_labels.json")
        #     file_labels = json.load(f)
        #     for file in file_labels:
        for i in range(1, 697):
            # if (file_labels[file] != reaction):
            #     continue
            # plot_count += 1

            try:
                # df = pd.read_csv("supervised_metric_learning/features_csvs/"+reaction+"/"+node+"_"+reaction+str(plot_count)+"_"+opponent+".csv")
                df = pd.read_csv("supervised_metric_learning/features_csvs/"+opponent+"/section" + str(i) + ".csv")
            except (FileNotFoundError):
                break

            d = dict(df.to_numpy())

            RMS     = float(d['RMS'])
            VAR     = float(d['VAR'])
            STD     = float(d['STD'])
            MEAN_f  = float(d['MEAN_f'])
            VAR_f   = float(d['VAR_f'])
            PtoP    = float(d['P2P'])
            CREST   = float(d['CREST FACTOR'])
            Q1_PWR  = float(d['Q1_Pwr_f'])
            Q2_PWR  = float(d['Q2_Pwr_f'])
            Q3_PWR  = float(d['Q3_Pwr_f'])
            HI_F_PWR = float(d['High_F_Pwr_f'])
            LO_F_PWR = float(d['Low_F_Pwr_f'])
            MED1_F_PWR = float(d['Med1_F_Pwr_f'])
            MED2_F_PWR = float(d['Med2_F_Pwr_f'])
            MAX_f   = float(d['MAX_f'])
            PEAK_f  = float(d['Highest_Peak_f'])
            POWER_f = HI_F_PWR + MED2_F_PWR + MED1_F_PWR + LO_F_PWR

            # if reaction == "Postgame":
            #     label = "Postgame"
            # else:
            #     label = "Moving"
            # label = reaction

            # if math.isnan(RMS): continue
            # if math.isnan(VAR): continue
            # if math.isnan(STD): continue
            if math.isnan(MEAN_f): continue
            # if math.isnan(VAR_f): continue
            if math.isnan(PtoP): continue
            # if math.isnan(CREST): continue
            if math.isnan(Q1_PWR): continue
            if math.isnan(Q2_PWR): continue
            if math.isnan(Q3_PWR): continue
            if math.isnan(HI_F_PWR): continue
            if math.isnan(LO_F_PWR): continue
            if math.isnan(MED1_F_PWR): continue
            if math.isnan(MED2_F_PWR): continue
            if math.isnan(MAX_f): continue
            if math.isnan(PEAK_f): continue
            if math.isnan(POWER_f): continue
                            
            # if (count % 3) == 0 or (count % 3) == 1 :
            #     Train_X[opponent].append([HI_F_PWR, MED1_F_PWR, MED2_F_PWR, LO_F_PWR])
            #     Train_Y[opponent].append(label)
            # else:
            #     Test_X[opponent].append([HI_F_PWR, MED1_F_PWR, MED2_F_PWR, LO_F_PWR])
            #     Test_Y[opponent].append(label)

            # count += 1

            X.append([LO_F_PWR, MED1_F_PWR, MED2_F_PWR, HI_F_PWR, Q1_PWR, Q2_PWR, Q3_PWR])
            # X.append([LO_F_PWR, MED1_F_PWR, MED2_F_PWR, HI_F_PWR, Q1_PWR, Q2_PWR, Q3_PWR, PEAK_f, MAX_f])
            # X.append([Q1_PWR, Q2_PWR, Q3_PWR,])
            I.append(i)

            # if (reaction == 'Booing'):
            #     Y.append('Cheering')
            #     continue
            # elif (reaction == 'Moving'):
            #     Y.append('Ugh')
            #     continue
            
            # Y.append(reaction)
                
            # print(reaction+" finished extracting")
        
        opp_X[opponent].append(X)
        opp_Y[opponent].append(Y)

        print("Beginning fit")

        # X_np = np.array(Train_X[opponent])
        X_np = np.array(opp_X[opponent][0])

        if opponent == 'OSU':
            start = time.time()
            model.fit(X_np)
            end = time.time()
            print("Took " + str(int(end - start)) + " sec")

        X_model = model.transform(X_np)

        # print(X_model.shape)

        # metric_func = model.get_metric()

        pairs = []

        # Form pairs with train set
        for x in X_model:
            pairs.append([x, X_model[-1]])

        scores = np.array(model.pair_score(pairs))
        distances = np.array(model.pair_distance(pairs))

        # print(np.min(scores), np.argmin(scores))

        # print(distances)

        idx = np.argpartition(scores, 10)
        print(idx[:10])
        for a in idx[:10]:
            print(a, scores[a])

        for s in scores:
            if s < -3:
                Y.append("Cheering")
            elif s < -1:
                Y.append("Ugh")
            else:
                Y.append("Postgame")

        colors = {'Booing': 'tomato',
            'Postgame': 'black',
            'Storming': 'orchid',
            'Ugh': 'blue',
            'Moving': 'yellow',
            'Cheering': 'green',
            'Unlabeled': 'gray'}

        Y_colors = []
        for val in Y:
            Y_colors.append(colors[val])

        plt.close()
        plt.scatter(I, scores * -1, c=Y_colors)
        plt.show()

        # plt.close()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(X_model[:,0], X_model[:,1], X_model[:,2])
        # plt.show()

        # Test set
        # for y in Test_X[opponent]:
        #     #form pairs
        #     pairs = []

        #     # Form pairs with train set
        #     for x in X_np:
        #         pairs.append([y, x])

        #     scores = np.array(model.pair_score(pairs))

        #     # scores = scores[(scores != 0)] # Remove exact match when testing with train data
            
        #     print(np.max(scores), np.argmax(scores))

        #     # print(opp_Y['OSU'][0][np.argmax(scores)])

        # pairs = []

        # # Form pairs with train set
        # for x in X_np:
        #     pairs.append([X_np[650], x])

        # scores = np.array(model.pair_score(pairs))

        # print(np.min(scores), np.argmin(scores))

        # idx = np.argpartition(scores, 5)
        # print(idx[:5])


