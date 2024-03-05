import re
import csv

def parse_and_filter_beta_decay(data):
    isotopes = []

    pattern = re.compile(r'^\s*(?P<MassNum>\d{3})\s+(?P<AtomicNum>\d{3})0\s+(?P<Element>\S{1,5})\s+'
                        r'(?P<MassExcess>[^ ]+)\s+(?P<BindingEnergy>[^ ]+)\s+(&?\s*)(?P<HalfLife>[^ ]+)\s+'
                        r'(?P<HalfLifeUnit>[^ ]+)\s+(?P<DecayModes>.+)$')

    def convert_to_seconds(half_life, unit):
        unit_conversion = {
            'ys': 1e-24, 'zs': 1e-21, 'as': 1e-18, 'fs': 1e-15,
            'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3,
            's': 1, 'm': 60, 'h': 3600, 'd': 86400, 'y': 31556926,
            'ky': 1e3 * 31556926, 'My': 1e6 * 31556926, 'Gy': 1e9 * 31556926,
            'Ty': 1e3 * 31556926, 'Py': 1e6 * 31556926, 'Ey': 1e9 * 31556926,
            'Zy': 1e3 * 31556926, 'Yy': 1e6 * 31556926
        }
        return half_life * unit_conversion.get(unit, -1)  # Default to -1 if unit not found

    for line in data:
        match = pattern.match(line)
        if match:
            decay_modes = match.group('DecayModes')
            if 'B-' in decay_modes:
                br_match = re.search(r'B-=(\d+\.?\d*)', decay_modes)
                if br_match:
                    branching_ratio = float(br_match.group(1)) / 100
                    half_life_str = match.group('HalfLife')
                    half_life_unit = match.group('HalfLifeUnit')
                    if '<' in half_life_str or '>' in half_life_str:
                        continue
                    
                    try:
                        half_life = float(half_life_str)
                        half_life_in_seconds = convert_to_seconds(half_life, half_life_unit)
                        partial_half_life = half_life_in_seconds / branching_ratio
                        
                        isotopes.append({
                            'Mass Number': match.group('MassNum'),
                            'Atomic Number': match.group('AtomicNum'),
                            'Element': match.group('Element'),
                            'Half-Life (s)': half_life_in_seconds,
                            'Beta Branch Ratio': branching_ratio,
                            'Beta Partial Half-Life (s)': partial_half_life
                        })
                    except ValueError:
                        continue
                    
    return isotopes

with open('Beta half life/nubase_4.mas20.txt') as file:
    data_lines = file.readlines()

filtered_isotopes = parse_and_filter_beta_decay(data_lines)

with open('Beta half life/Beta_Half_Lives.csv', 'w', newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=['Mass Number', 'Atomic Number', 'Element', 
                                                     'Half-Life (s)', 'Beta Branch Ratio', 'Beta Partial Half-Life (s)'])
    writer.writeheader()
    for isotope in filtered_isotopes:
        writer.writerow(isotope)

print("Processing complete. Data written to Beta_Half_Lives.csv")