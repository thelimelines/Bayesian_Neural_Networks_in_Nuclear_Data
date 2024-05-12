import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Load the data
file_path = 'Final submission/Predicted_Beta_Half_Lives.csv'
data = pd.read_csv(file_path)

# Calculate neutron number
data['Neutron Number (N)'] = data['Mass Number (A)'] - data['Atomic Number (Z)']

# Function to strip numbers and extract element name
def get_element_name(full_name):
    if isinstance(full_name, str):
        return ''.join([i for i in full_name if not i.isdigit()])
    return "Unknown"  # Return a placeholder if the data is not a string

# Initialize the root GUI
root = tk.Tk()
root.title("Constant Neutron Data Plotter")

# Setup the initial figure and canvas
fig = Figure(figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_final_element_data_with_constant_neutrons(N):
    N = int(N)  # Ensure N is an integer
    element_data = data[data['Neutron Number (N)'] == N]
    if element_data.empty:
        fig.clf()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No data available for this neutron number', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("No Data")
        canvas.draw()
        return

    # Clear previous figure to avoid overlap and redraw
    fig.clf()
    plt.rcParams.update({'font.size': 32})
    ax = fig.add_subplot(111)
    
    element_name = get_element_name(element_data['Element'].iloc[0])
    ax.errorbar(element_data['Atomic Number (Z)'], element_data['Predicted Half-Life (log(s))'],
                 yerr=[element_data['Predicted Half-Life (log(s))'] - element_data['Lower CI (95%)'],
                       element_data['Upper CI (95%)'] - element_data['Predicted Half-Life (log(s))']],
                 fmt='x', color='blue', ecolor='darkgray', elinewidth=4, capsize=10, markersize=20, label='Predicted with 95% CI', zorder=1)
    ax.scatter(element_data['Atomic Number (Z)'], element_data['Beta Partial Half-Life (log(s))'], color='red', label='Experimental', s=100, zorder=2)

    ax.set_title(f'Neutron Number {N}: Beta Partial Half-Lives with 95% CI (log scale)')
    ax.set_xlabel('Atomic Number (Z)')
    ax.set_ylabel('Log Half-Life (s)')
    ax.legend()
    ax.grid(True)

    # Invert the X-axis to flip the atomic number axis
    ax.invert_xaxis()

    # Define the r-process zones for various neutron numbers
    r_process_zones = {
        82: (42, 49),
        126: (61, 72),
        60: (33, 35),
        62: (33, 36),
        64: (34, 37),
        66: (35, 38),
        68: (36, 40),
        69: (38.5, 39.5),
        70: (38, 40),
        72: (39, 41),
        74: (39, 41),
        76: (39, 42),
        78: (40, 42),
        80: (41, 44)
    }

    # Apply r-process zones if defined for this neutron number
    if N in r_process_zones:
        start_z, end_z = r_process_zones[N]
        ax.axvspan(start_z, end_z, color='yellow', alpha=0.3)
        mid_z = (start_z + end_z) / 2
        ax.text(mid_z, ax.get_ylim()[1], 'r-process zone     ', horizontalalignment='center', verticalalignment='top', fontsize=32, color='black', rotation=90)
    
    canvas.draw()


# Create a slider for neutron number with dynamic update
scale = tk.Scale(root, from_=data['Neutron Number (N)'].min(), to=data['Neutron Number (N)'].max(), orient=tk.HORIZONTAL, command=plot_final_element_data_with_constant_neutrons)
scale.pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()