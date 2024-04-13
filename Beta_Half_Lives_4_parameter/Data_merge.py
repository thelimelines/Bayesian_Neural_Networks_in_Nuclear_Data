import pandas as pd

def merge_datasets(beta_half_lives_path, q_values_path, output_path):
    # Load data from the specified file paths
    beta_half_lives = pd.read_csv(beta_half_lives_path)
    q_values = pd.read_csv(q_values_path)

    # Rename columns to ensure matching for the merge
    q_values.rename(columns={"A": "Mass Number (A)", "Z": "Atomic Number (Z)"}, inplace=True)

    # Perform an outer join to include all records, filling unmatched entries with NaN
    merged_data = pd.merge(beta_half_lives, q_values, on=['Atomic Number (Z)', 'Mass Number (A)'], how='outer')

    # Remove columns not needed
    merged_data.drop(['Mass excess WS4 (keV)', 'Mass excess WS4+RBF (keV)'], axis=1, inplace=True)

    # Save the final merged data to the specified output path
    merged_data.to_csv(output_path, index=False)

# Define file paths
beta_half_lives_path = 'Beta_Half_Lives_4_parameter/Log_Beta_Half_Lives.csv'
q_values_path = 'Beta_Half_Lives_4_parameter/Q_Processed_WS4.csv'
output_path = 'Beta_Half_Lives_4_parameter/Training_data.csv'

# Call the function with the file paths
merge_datasets(beta_half_lives_path, q_values_path, output_path)
