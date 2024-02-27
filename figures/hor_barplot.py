import matplotlib.pyplot as plt
import os 
import numpy as np

# Smartphone names
smartphones = ['Pixel 6a', 'Samsung S10', 'Poco M5', 'Pixel 1 XL', 'Moto g31'] #'Pixel 8', 

release_year = [2022, 2019, 2022, 2016, 2021]

# Corresponding training times (in hours, for example)
training_times = np.array([89.78, 112.6, 120.2, 178.184, 197.166]) #92.30, 

# to days
training_times = training_times*1500
training_times = (training_times/3600)

# Calculate 10% of each training time for the initial portion of the bars
initial_portion = training_times * 0.1

# Create a horizontal bar plot
plt.figure(figsize=(30, 6))  # Adjust the size as needed

# Plot the full bars
plt.barh(smartphones, training_times, color='skyblue')

# Overlay the 10% initial portions with a different color
plt.barh(smartphones, initial_portion, color='orange')

plt.xlabel('Training Time (hours), Med-NCA 256x256')
plt.title('Training Time on Different Smartphones')
plt.grid(axis='x')  # Add grid lines for better readability

# Save the plot as a PDF
plt.savefig(os.path.join('/home/jkalkhof_locale/Downloads/Figures/', 'training_times_on_smartphones.pdf'), format='pdf')

# Optionally display the plot
plt.show()