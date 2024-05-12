import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the updated data from the CSV file
file_path = 'Final submission\\Predicted_Beta_Half_Lives.csv'
data = pd.read_csv(file_path)

# Convert predicted half-lives from log scale to linear scale for visualization
data['Predicted Beta Half-Life Linear'] = np.power(10, data['Predicted Half-Life (log(s))'])
data['Experimental Beta Half-Life Linear'] = np.power(10, data['Beta Partial Half-Life (log(s))'])

# Convert lower and upper confidence intervals from log scale to linear scale for visualization
data['Lower CI (95%) Linear'] = np.power(10, data['Lower CI (95%)'])
data['Upper CI (95%) Linear'] = np.power(10, data['Upper CI (95%)'])

# Sample approximately 1/10th of the data randomly
# sampled_data = data.sample(frac=1, random_state=3141592) #uncomment for full plot

# Filter data for half-lives less than 10^6 seconds
filtered_data = data[data['Experimental Beta Half-Life Linear'] < 1e6]
sampled_data = filtered_data.sample(frac=1, random_state=908)  # Adjust fraction as necessary depending on data size


# Scatter plot with error bars for the sampled data
plt.figure(figsize=(12, 10))
plt.rcParams.update({'font.size': 16})
plt.errorbar(sampled_data['Experimental Beta Half-Life Linear'], sampled_data['Predicted Beta Half-Life Linear'],
             yerr=[sampled_data['Predicted Beta Half-Life Linear'] - sampled_data['Lower CI (95%) Linear'],
                   sampled_data['Upper CI (95%) Linear'] - sampled_data['Predicted Beta Half-Life Linear']],
             fmt='o', ecolor='gray', alpha=0.7, label='95% Confidence Interval',
             color='darkblue', markersize=5, elinewidth=2, capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Experimental Beta Partial Half-Life (s)')
plt.ylabel('Predicted Beta Half-Life (s)')
plt.title('Sampled Experimental vs. Predicted Beta Half-Lives with 95% Confidence Intervals')

# Adding line of equality
plt.plot([sampled_data['Experimental Beta Half-Life Linear'].min(), sampled_data['Experimental Beta Half-Life Linear'].max()], 
         [sampled_data['Experimental Beta Half-Life Linear'].min(), sampled_data['Experimental Beta Half-Life Linear'].max()], 
         'r--', label='Predicted = Experimental')

# Adjust grid for better visibility with lighter subgrids
plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')

plt.legend()
plt.show()
