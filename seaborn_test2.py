import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
from matplotlib import ticker as mticker
import math 

#prostate_pred.csv

d = {}

hippocampus = True

if hippocampus:
    df = pd.read_csv(r'C:\Users\John\Desktop\hippocampus_pred.csv', sep="\t")
else:
    print("PROSTATE")
    df = pd.read_csv(r'C:\Users\John\Desktop\prostate_pred.csv', sep="\t")
    #d['Method'] = ["UNet", "UNet", "UNet", "UNet", "UNet", "UNet", "UNet", "UNet", "UNet", "UNet2", "UNet2", "UNet2", "UNet2", "UNet2", "UNet2", "UNet2", "UNet2", "UNet2"]
    #d['Dice'] = [0.5429356098175049, 0.70166015625, 0.6451302170753479, 0.5671334266662598, 0.5238515138626099, 0.42959779500961304, 0.46223974227905273, 0.5022304654121399, 0.49402832984924316, 0.5429356098175049, 0.70166015625, 0.6451302170753479, 0.5671334266662598, 0.5238515138626099, 0.42959779500961304, 0.46223974227905273, 0.5022304654121399, 0.49402832984924316]
    #d['STDD'] = [0.2]
    #d['Parameters'] = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]

df.astype({'Parameters': 'int32'}).dtypes
#df.astype({'Dice': 'int32'}).dtypes
#df['Parameters'] = np.log10(df['Parameters'])

#df = pd.DataFrame(data=d)

params = df['Parameters']

unique_params = np.unique(np.log(df['Parameters']))
if False:
    #for i in range(len(unique_params)):
    #    val = unique_params[i]
    #    pos = 0
    #    while True:
    #        val = val / 10
    #        pos = pos+1
    #        if val < 10:
    #            break
    #    print(pos, val)
    #    pos = pos + math.log(val)
    #    print(pos)
    #    unique_params[i] = pos
    print(unique_params)
    for i in range(len(unique_params)):
        if unique_params[i] > 14:
            unique_params[i] = unique_params[i] -4

    ax = df.boxplot(column=['Dice'], by=['Parameters'], positions=unique_params)
else:
    print(unique_params)

    #violinplot
    #ax = df.boxplot(column=['Dice'], by=['Parameters'], positions=unique_params)#
    sns.set(rc={'figure.figsize':(10, 2.4)})
    #
    #ax = sns.boxenplot(data=df, x='Parameters', y='Dice', hue='Method', dodge=False)

    df = df.sort_values(by=['Parameters'], ascending=True)
    
    #df_melt.sort_values(by=['Method'])

    #groupedvalues = df.groupby('Parameters').sum().reset_index()
    #for index, row in groupedvalues.iterrows():
    #    g.text(row.name, row.tip, round(row.total_bill, 2), color='black', ha="center")

    #print(df)
    if True:
        print(np.unique(df['Dice']))
        ax = sns.violinplot(data=df, x='Parameters', y='Dice', hue='Method', dodge=False)#, errorbar="STDD") #, cut=0
        #ax.set_ylim(0,1)
    else:
        if False:
            print(df)
            #df['Parameters'] = np.log10(df['Parameters']) #/ np.max(df['Parameters']) #np.log(df['Parameters'])/
            df['Parameters'][df['Parameters'] > 5000000] = 5000000
            df['Parameters'] = df['Parameters'] / np.max(df['Parameters'])
            df_melt = pd.melt(df, id_vars="Method", var_name="Dice", value_name="Score")
            ax = sns.scatterplot(data=df_melt, x='Method', y='Score', hue="Dice")#, hue='Method') #, order=df_melt.sort_values('Score')
        else:
            #df = df.mean(axis=0)
            ax = sns.scatterplot(data=df, x='Parameters', y='Dice')#, size="Parameters")
            ax.set_xscale('log')
    #print(params)
    #ax.set_xscale('log')



#ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
#xmin, xmax = ax.get_xlim()
#tick_range = np.arange(np.floor(xmin), xmax)
#ax.xaxis.set_ticks(tick_range)
#ax.xaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)

#ax.set_xscale('log')

plt.show()