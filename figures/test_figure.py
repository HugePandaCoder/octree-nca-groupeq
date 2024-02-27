#%%
import numpy as np
import matplotlib.pyplot as plt

# Assuming 5 models, 3 datasets, and the same model parameters for simplicity
model_params = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
performance = np.random.rand(3, 3, 5)  # Random performance data: [train_dataset, eval_dataset, model]
std_dev = np.random.rand(3, 3, 5) * 0.05  # Random std deviations

datasets = ['Dataset A', 'Dataset B', 'Dataset C']
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

fig, axes = plt.subplots(3, 1, figsize=(8, 18), sharex=True)  # 3 subplots for 3 training datasets, stacked vertically

for i, ax in enumerate(axes):
    for j, color in enumerate(colors):
        ax.errorbar(model_params, performance[i, j], yerr=std_dev[i, j], fmt=markers[j], color=color,
                    label=f'Evaluated on {datasets[j]}')
    ax.set_title(f'Trained on {datasets[i]}')
    ax.set_xscale('log')
    if i == 2:  # Only add the xlabel to the bottom subplot
        ax.set_xlabel('Model Parameters')
    ax.grid(True)
    if i == 1:  # Adding ylabel to the middle subplot for balance
        ax.set_ylabel('Performance Metric')

axes[1].legend(title='Evaluation Dataset', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.tight_layout()
plt.show()

# %%
