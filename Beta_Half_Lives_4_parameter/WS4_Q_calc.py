import pandas as pd

file_path = 'Beta_Half_Lives_4_parameter/WS4_RBF.txt'
output_csv_path = 'Beta_Half_Lives_4_parameter/Q_Processed_WS4.csv'

# Load data into a DataFrame
data = pd.read_csv(file_path, skiprows=15, delim_whitespace=True, encoding='ISO-8859-1',
                   names=['A', 'Z', 'WS4', 'WS4+RBF'], dtype={'A': int, 'Z': int, 'WS4': float, 'WS4+RBF': float})

# Convert mass excess from MeV to keV
data['WS4'] *= 1000
data['WS4+RBF'] *= 1000

# Create a dictionary to easily look up by (A, Z)
lookup = {(row['A'], row['Z']): {'WS4': row['WS4'], 'WS4+RBF': row['WS4+RBF']} for index, row in data.iterrows()}

# Prepare to write output
output_data = []

# Process each row to calculate Q_beta- values
for index, row in data.iterrows():
    A = row['A']
    Z = row['Z']
    current_ws4 = row['WS4']
    current_ws4_rbf = row['WS4+RBF']
    
    # Look up the daughter nucleus (A, Z+1)
    daughter_data = lookup.get((A, Z + 1))
    
    if daughter_data:  # Make sure the daughter nucleus exists
        # Calculate Q_beta- values
        q_beta_ws4 = current_ws4 - daughter_data['WS4']
        q_beta_ws4_rbf = current_ws4_rbf - daughter_data['WS4+RBF']
        
        # Append the result to the output data list
        output_data.append([A, Z, current_ws4, current_ws4_rbf, q_beta_ws4, q_beta_ws4_rbf])
    else:
        # Append without Q values if daughter not found
        output_data.append([A, Z, current_ws4, current_ws4_rbf, None, None])

# Convert output data to DataFrame
output_df = pd.DataFrame(output_data, columns=['A', 'Z', 'Mass excess WS4 (keV)', 'Mass excess WS4+RBF (keV)', 
                                               'Q_beta- WS4 (keV)', 'Q_beta- WS4+RBF (keV)'])

# Save to CSV
output_df.to_csv(output_csv_path, index=False)

print("Data processed and saved to CSV successfully.")