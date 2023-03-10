import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np

hippocampus = False
dots = True
inverseDots = False
val = 0.8

if hippocampus:
    if not inverseDots:
        df = pd.read_csv(r'C:\Users\John\Desktop\OOD_Detection.csv', sep="\t")
        df_thres = pd.read_csv(r'C:\Users\John\Desktop\OOD_Hipp_train.csv', sep="\t")
    else: 
        df_thres = pd.read_csv(r'C:\Users\John\Desktop\OOD_Detection.csv', sep="\t")
        df = pd.read_csv(r'C:\Users\John\Desktop\OOD_Hipp_train.csv', sep="\t")
else:
    if not inverseDots:
        df = pd.read_csv(r'C:\Users\John\Desktop\OOD_Detection_Prostate.csv', sep="\t")
        df_thres = pd.read_csv(r'C:\Users\John\Desktop\OOD_Prost_train.csv', sep="\t")
    else: 
        df_thres = pd.read_csv(r'C:\Users\John\Desktop\OOD_Detection_Prostate.csv', sep="\t")
        df = pd.read_csv(r'C:\Users\John\Desktop\OOD_Prost_train.csv', sep="\t")
    
#df = pd.DataFrame(data=data).T
#df = df2
#figure, axes = plt.subplots(2, 1)
#index = 0

df["Metric"] = df["Metric"].clip(0, 1)
df["Dice"] = df["Dice"].clip(0, 1)


df_thres["Metric"] = df_thres["Metric"].clip(0, 1)
df_thres["Dice"] = df_thres["Dice"].clip(0, 1)

print(df.columns)
#df = df[df.Augmentation != "Ghosting"]


if True:
    df_none = df_thres
    df_none_inv = df
else:
    df_none = df[(df.Augmentation == "None") | (df.Augmentation == "Noise") ]
    df_none_inv = df[(df.Augmentation == "Ghosting") | (df.Augmentation == "Spike") ]

#df_none_Inv = df[(df.Augmentation == "Noise") | (df.Augmentation == "Ghosting") ] 
#score = df_none["Dice"] / df_none["Metric"]
#print(sum(df_none["Dice"]), sum(1 - df_none["Metric"]))
#score = sum(df_none["Dice"]) / sum(1 - df_none["Metric"])
#print("SCORE", score, score * 0.8)

regr = linear_model.LinearRegression()

metric = np.matrix(df_none["Metric"]).T
dice = np.matrix(df_none["Dice"]).T

#regr.fit(metric, dice)
regr.fit(dice, metric)


#print(regr.score(metric, dice)) #([[0.8]]))
avg_distance = sum(abs(regr.predict(metric)-dice)) / len(dice)
pos = regr.predict([[val]])#np.matrix(val).T)

print("REGR 1", val, regr.predict([[val]]))



def tp_tn(dfl, metric, predict, threshold_x, threshold_y):
    dfl['Right'] = 0
    for index, row in dfl.iterrows():
        if row['Metric'] < threshold_x and row['Dice'] > threshold_y:# - avg_distance:
            #dfl[index]['Right'] = True
            dfl.loc[index, 'Right'] = 1
        if row['Metric'] > threshold_x and row['Dice'] < threshold_y:# - avg_distance:
            dfl.loc[index, 'Right'] = 1
    return dfl

tp_tn = tp_tn(df_none_inv, metric, regr.predict(metric), threshold_x = pos, threshold_y = val)

print(np.unique(tp_tn['Right']))
tp = np.array(tp_tn['Right'])==1
tp = sum(tp)
tn = np.array(tp_tn['Right'])==2
tn = sum(tn)

#mask_t = sum(tp_tn['Right'])

# HERE
print("TP / TN",tp, tn, len(tp_tn['Right']))


#print("TP / TN", len(mask) / len(tp_tn['Right']))

#df = tp_tn(df, metric, regr.predict(metric), threshold_x = pos, threshold_y = val)

# ADD MArkers

def add_markers(dfl):
    dfl['Markers'] = "o"
    #for index, row in dfl.iterrows():
    #    if row['Augmentation'] is 'None':
    #        dfl.loc[index, 'Markers'] = "v"
    #    else:
    #        dfl.loc[index, 'Markers'] = "o"

    print(dfl['Markers'])
    return dfl

df = add_markers(df)



print(df)

#f, ax = plt.subplots(figsize=(6, 6))
#sns.scatterplot(data = df, x="Metric", y="Dice", s=5, color=".15")
#sns.histplot(data = df, x="Metric", y="Dice", bins=50, pthresh=.1, cmap="mako")
#sns.kdeplot(data = df, x="Metric", y="Dice", levels=5, color="w", linewidths=1)
#plt.show()

#exit()

#print(regr.predict(metric))
#print(metric[0])

#print(metric)
x_pos = sum(df["Metric"]) / len(df["Metric"])
y_pos = sum(df["Dice"]) / len(df["Dice"])


if dots:
    markers = {}
    if not inverseDots:
        markers["None"] = "P"
        markers["Noise"] = "X"
        markers["Spike"] = "o"
        markers["Ghosting"] = "s"
    else:
        markers["None"] = "v"
        markers["Noise"] = "^"

    #ax = sns.regplot(data = df, x = "Metric", y = "Dice")
    ax2 = sns.scatterplot(data = df, x = "Metric", y = "Dice", style="Augmentation", hue="Right", markers=markers)#, ax=axes[0] )
    avg_distance = sum(abs(regr.predict(metric)-dice)) / len(dice)
    #print(sum(abs(regr.predict(metric)-dice)), regr.predict(metric)-dice, dice, len(dice))
    #print(avg_distance)
    plt.axvline(x=pos)
    plt.plot((0,1), (val, val))
    #print("TEST", avg_distance)
    #plt.fill_between(x=(0,1), y1=np.ravel((val, val)-avg_distance), y2=np.ravel((val, val)+avg_distance), alpha=.5)
    
    #plt.axhline(y=val)
    #plt.axhline(y=val+avg_distance)
    #plt.axhline(y=val-avg_distance)
    #sns.lineplot(x=metric, y= )

    metric = [[a] for a  in np.arange(0, 10, 1)]
    #metric = [[0.1], [0.2]]
    print(metric)
    plt.plot(regr.predict(metric), metric)
    
    #plt.fill_between(x=np.ravel(metric), y1=np.ravel(regr.predict(metric)-avg_distance), y2=np.ravel(regr.predict(metric)+avg_distance), alpha=.5)
    plt.fill_between(x=np.ravel(regr.predict(metric)), y1=np.ravel(metric-avg_distance), y2=np.ravel(metric+avg_distance), alpha=.5)
    
    #plt.plot(metric, regr.predict(metric)-avg_distance)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)

    print("REGR 2", val, regr.predict([[val]]))
    

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
else:
    sns.jointplot(data = tp_tn, x = "Metric", y = "Dice", hue="Right", xlim = (0,1), ylim=(0,1), marginal_ticks=True )
#plt.plot(metric, regr.predict(metric))
#plt.plot(metric, regr.predict(metric)+0.1)
#plt.plot(metric, regr.predict(metric)-0.1)

#plt.plot((0, 1), (y_pos, y_pos * 2), color='blue', linewidth=3)
#plt.axhline(y=15) 
#plt.xticks(())
#plt.yticks(())
#axes[0].set_xscale('log')
#axes[0].set_title("Hippocampus OOD Detection")
#plt.savefig(r'M:\Seaborn\MICCAI\prostate_certainty.svg')
plt.show()



sns.scatterplot(data = df2, x = "Metric", y = "Dice", hue="Augmentation", ax=axes[1] )
#fig2 = sns.jointplot(data = df2, x = "Metric", y = "Dice", hue="Augmentation", ax=axes[1], xlim = (0,1), ylim=(0,1) )

#ax.set_xlim(0, 0.7)
axes[1].set_xscale('log')
axes[1].set_title("Prostate OOD Detection")

plt.show()