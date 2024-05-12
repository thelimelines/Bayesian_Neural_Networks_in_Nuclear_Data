import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Load the data
file_path = 'Final submission/Predicted_Beta_Half_Lives.csv'
data = pd.read_csv(file_path)

# Function to strip numbers and extract element name
def get_element_name(full_name):
    if isinstance(full_name, str):
        return ''.join([i for i in full_name if not i.isdigit()])
    return "Unknown"  # Return a placeholder if the data is not a string

# Initialize the root GUI
root = tk.Tk()
root.title("Element Data Plotter")

# Setup the initial figure and canvas
fig = Figure(figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_final_element_data_with_ci(Z):
    Z = int(Z)  # Ensure Z is an integer
    element_data = data[data['Atomic Number (Z)'] == Z]
    if element_data.empty:
        fig.clf()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No data available for this atomic number', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("No Data")
        canvas.draw()
        return

    # Clear previous figure to avoid overlap and redraw
    fig.clf()
    plt.rcParams.update({'font.size': 32})
    ax = fig.add_subplot(111)
    
    element_name = get_element_name(element_data['Element'].iloc[0])
    ax.errorbar(element_data['Mass Number (A)'], element_data['Predicted Half-Life (log(s))'],
                 yerr=[element_data['Predicted Half-Life (log(s))'] - element_data['Lower CI (95%)'],
                       element_data['Upper CI (95%)'] - element_data['Predicted Half-Life (log(s))']],
                 fmt='x', color='blue', ecolor='darkgray', elinewidth=4, capsize=10, markersize=20, label='Predicted with 95% CI',zorder=1)
    ax.scatter(element_data['Mass Number (A)'], element_data['Beta Partial Half-Life (log(s))'], color='red', label='Experimental',s=100,zorder=2)

    ax.set_title(f'{element_name} Beta Partial Half-Lives with 95% CI (log scale)')
    ax.set_xlabel('Mass Number (A)')
    ax.set_ylabel('Log Half-Life (s)')
    ax.legend()
    ax.grid(True)
    
    canvas.draw()

# Create a slider for atomic number with dynamic update
scale = tk.Scale(root, from_=data['Atomic Number (Z)'].min(), to=data['Atomic Number (Z)'].max(), orient=tk.HORIZONTAL, command=plot_final_element_data_with_ci)
scale.pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()