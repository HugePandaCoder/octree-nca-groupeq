import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
from matplotlib import ticker as mticker
import math 

#prostate_pred.csv

df = pd.read_csv(r'C:\Users\John\Desktop\rasp_times.csv', sep="\t")

df['Time'] = df['Time']/60

print(df)

sns.set(rc={'figure.figsize':(10, 2.4)})
a = sns.scatterplot(data=df, x='Voxels', y='Time')
b = sns.lineplot(data=df, x='Voxels', y='Time')

a.set(xlim=(0, None), ylim=(0, None))
b.set(xlim=(0, None), ylim=(0, None))

plt.show()