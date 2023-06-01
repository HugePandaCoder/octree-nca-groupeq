#%%

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kernel_size = [3, 5, 7, 9]
dice = [78.62, 75.22, 75.88, 76.09]
deviation = [1.58, 1.45, 1.59, 1.55]

sns.set_theme(style="whitegrid")

df = pd.DataFrame(list(zip(kernel_size, dice, deviation)), columns= ['Kernel Size', 'Dice', 'Deviation'])

ax = sns.scatterplot(data=df, x="Kernel Size", y="Dice") #, errorbar="Deviation")
ax.set(ylim=(50, 90))

ax.axhline(84.59)
ax.axhline(68.56)

plt.savefig('/home/jkalkhof_locale/Documents/temp/NeurIps/kernelk_size.svg')
plt.show()

kernel_size = [10, 30, 50, 90]
dice = [80.17, 75.88, 67.83, 65.67]
deviation = [1.59, 1.59, 1.78, 1.9]

sns.set_theme(style="whitegrid")

df = pd.DataFrame(list(zip(kernel_size, dice, deviation)), columns= ['Steps', 'Dice', 'Deviation'])

ax = sns.scatterplot(data=df, x="Steps", y="Dice") #, errorbar="Deviation")
ax.set(ylim=(50, 90))

ax.axhline(84.59)
ax.axhline(68.56)

plt.savefig('/home/jkalkhof_locale/Documents/temp/NeurIps/step_size.svg')
plt.show()
# %%
