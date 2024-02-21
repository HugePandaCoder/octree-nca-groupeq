import matplotlib.pyplot as plt
import os 

# Smartphone names
smartphones = ['High-End', '1', '2', '3', 'Low End']

# Corresponding training times (in hours, for example)
training_times = [15.6, 20.3, 50.7, 113.4, 140.1]

# Create a horizontal bar plot
plt.figure(figsize=(30, 6))  # Adjust the size as needed
plt.barh(smartphones, training_times, color='skyblue')
plt.xlabel('Training Time (second per batch), Med-NCA 256x256')
plt.title('Training Time on Different Smartphones')
plt.grid(axis='x')  # Add grid lines for better readability

# Save the plot as a PDF
plt.savefig(os.path.join('/home/jkalkhof_locale/Downloads/Figures/', 'training_times_on_smartphones.pdf'), format='pdf')

# Optionally display the plot
plt.show()
