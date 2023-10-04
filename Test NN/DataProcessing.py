import csv

# Function to parse a single line of data
def parse_line(line):
    neutron_number = int(line[6:9].strip())
    proton_number = int(line[11:14].strip())
    atomic_number = int(line[16:19].strip())
    element_name = line[20:22].strip()
    binding_energy_str = line[56:67].strip()
    uncertainty_str = line[69:78].strip()
    
    # Only remove rows where '#' appears in the binding energy or uncertainty fields
    if '#' in binding_energy_str or '#' in uncertainty_str:
        return None
    try:
        binding_energy_per_nucleon = float(binding_energy_str)
        binding_energy_uncertainty = float(uncertainty_str)
    except ValueError:  # Handle non-numeric values
        binding_energy_per_nucleon = None
        binding_energy_uncertainty = None
    return neutron_number, proton_number, atomic_number, element_name, binding_energy_per_nucleon, binding_energy_uncertainty

# Initialize list to hold parsed data
parsed_data = []

# Read and parse the file
with open('Test NN\AME2020_1.txt', 'r') as file:
    # Skip the first 36 lines of preamble
    for _ in range(36):
        file.readline()
    # Parse the remaining lines
    for line in file:
        parsed_line = parse_line(line)
        if parsed_line is not None:
            parsed_data.append(parsed_line)

# Create the CSV file
csv_file_path = 'Test NN\AME2020_converted.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Neutron Number', 'Proton Number', 'Atomic Number', 'Element Name', 'Binding Energy per Nucleon (keV)', 'Uncertainty (keV)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # Write headers
    writer.writeheader()
    # Write data
    for neutron_number, proton_number, atomic_number, element_name, binding_energy_per_nucleon, binding_energy_uncertainty in parsed_data:
        writer.writerow({
            'Neutron Number': neutron_number,
            'Proton Number': proton_number,
            'Atomic Number': atomic_number,
            'Element Name': element_name,
            'Binding Energy per Nucleon (keV)': binding_energy_per_nucleon,
            'Uncertainty (keV)': binding_energy_uncertainty
        })

# Number of lines parsed
print("Parsed ",len(parsed_data)," lines to ", csv_file_path)