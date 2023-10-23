# %% 
import matplotlib.pyplot as plt

# Data
x = ['text1', 'text2', 'text3']
y = [3, 5, 4]  # Placeholder y-values for the bars
errors = [0.2, 0.3, 0.2]  # Placeholder errors for the bars

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Use a gray background for the plot
ax.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#f5f5f5')

# Create bars with error bars
bars = ax.bar(x, y, yerr=errors, capsize=5, color=['#1f77b4', '#ff7f0e', '#d62728'], edgecolor='black')

# Customize patterns of the bars
patterns = ['/', '||', '\\\\']
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

# Enhance y axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

ax.tick_params(axis='both', which='both', length=0)
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

# Add labels and title
ax.set_ylabel('Y-axis label', fontsize=14, fontweight='bold')
ax.set_xlabel('X-axis label', fontsize=14, fontweight='bold')
ax.set_title('Professional Bar Chart', fontsize=16, fontweight='bold', pad=20)

# Set font size for tick labels
ax.tick_params(axis='both', labelsize=12)

# Display the figure
plt.tight_layout()
plt.show()



if False:
        import matplotlib.pyplot as plt
        import numpy as np

        # Placeholder data
        labels = ['Bad', 'Meh', 'Okay', 'Good', 'Great']
        sizes = [1, 1, 4, 1, 1]  # equal distribution for now, you can adjust as needed
        colors = ['#f06767', '#f7d07b', '#8ed2c9', '#6597cb', '#435f7a']
        explode = (0.1, 0, 0, 0, 0)  # explode the first slice for emphasis

        # Plotting
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%',
                startangle=90, pctdistance=0.85, explode=explode)

        # Draw a circle in the center for 'donut' style
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # Equal aspect ratio ensures pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        plt.show()
# %%
