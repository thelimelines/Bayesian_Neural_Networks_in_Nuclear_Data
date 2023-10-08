# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading

# Define custom callback for real-time plotting
class RealTimePlot(Callback):
    def __init__(self, ax):
        self.ax = ax

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.ax.clear()

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.ax.clear()
        self.ax.plot(self.losses, label='loss')
        self.ax.plot(self.val_losses, label='val_loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss Value')
        self.ax.legend()
        self.ax.figure.canvas.draw()

# GUI to allow user to test model on real data
def predict_binding_energy():
    neutron_number, proton_number = float(neutron_entry.get()), float(proton_entry.get())
    input_data = scaler_x.transform(np.array([[neutron_number, proton_number]])) # scales user input to model inputs
    prediction = scaler_y.inverse_transform(model.predict(input_data)) # scales model output to real output (keV)
    in_training_set = any(np.all(X_train == input_data, axis=1)) # checks if the user input was used in training
    actual_value = df.loc[(df['Neutron Number'] == neutron_number) & (df['Proton Number'] == proton_number)] # retrieves real value
    actual_value = actual_value['Binding Energy per Nucleon (keV)'].values[0] if not actual_value.empty else 'N/A' # string manipulation for UI
    result_label.config(text=f"In Training Set: {in_training_set}\nActual Value: {actual_value}\nPredicted Value: {prediction[0][0]:.2f}")

# Function to render the 3D plot with given neutron and proton grids and predicted energies
def update_plot(neutron_grid, proton_grid, predicted_energy):
    ax.clear()
    canvas_3dplot = FigureCanvasTkAgg(fig, master=root)
    
    # Plot the neural network prediction in red
    ax.plot_surface(neutron_grid, proton_grid, predicted_energy.reshape(neutron_grid.shape), 
               color='r', alpha=0.6)
    
    # Plot the real values directly from the dataset in black
    real_neutrons = df['Neutron Number'].values
    real_protons = df['Proton Number'].values
    real_energies = df['Binding Energy per Nucleon (keV)'].values
    ax.scatter(real_neutrons, real_protons, real_energies, color='black', alpha=0.1)
    
    # Align the axes at (0,0,0)
    ax.set_xlim3d([0, neutron_grid.max()])
    ax.set_ylim3d([0, proton_grid.max()])
    ax.set_zlim3d([0, max(predicted_energy.max(), np.nanmax(real_energies))])
    
    ax.set_title('NN Prediction vs Real Data')
    ax.set_xlabel('Neutron Number')
    ax.set_ylabel('Proton Number')
    ax.set_zlabel('Binding Energy per Nucleon (keV)')
    canvas_3dplot.draw()

# Function to prepare the data and call update_plot for rendering
def update_3D_plot():
    # Generate a grid of neutron and proton numbers based on the dataset range
    neutron_numbers = np.linspace(min(df['Neutron Number']), max(df['Neutron Number']), 50)
    proton_numbers = np.linspace(min(df['Proton Number']), max(df['Proton Number']), 50)
    neutron_grid, proton_grid = np.meshgrid(neutron_numbers, proton_numbers)

    # Preparing the data for NN prediction
    grid_data = np.array([neutron_grid.flatten(), proton_grid.flatten()]).T
    scaled_grid_data = scaler_x.transform(grid_data)

    # Predict binding energy using the trained NN
    predicted_energy = model.predict(scaled_grid_data)
    predicted_energy = scaler_y.inverse_transform(predicted_energy).flatten()

    # Call the existing update_plot function
    update_plot(neutron_grid, proton_grid, predicted_energy)

    # Use draw_idle to reduce jitter
    canvas_3dplot.draw_idle()

# Load and preprocess data
df = pd.read_csv('Test NN\\AME2020_converted.csv') # dataframe retrieved from trimmed AME

# X and Y are the inputs and outputs of the model respectively, note X is an array size 2
X, y = df[['Neutron Number', 'Proton Number']].values, df['Binding Energy per Nucleon (keV)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #30% testing, 70%training, random_State is a seed set for reproducability

scaler_x, scaler_y = StandardScaler().fit(X_train), StandardScaler().fit(y_train.reshape(-1, 1))
X_train, X_test = scaler_x.transform(X_train), scaler_x.transform(X_test)
y_train, y_test = scaler_y.transform(y_train.reshape(-1, 1)).flatten(), scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# GUI setup
root = tk.Tk()
root.title('Neural Network Prediction for Nuclear Binding Energy')

neutron_label, neutron_entry = ttk.Label(root, text='Neutron Number:'), ttk.Entry(root)
neutron_label.grid(row=0, column=0), neutron_entry.grid(row=0, column=1)

proton_label, proton_entry = ttk.Label(root, text='Proton Number:'), ttk.Entry(root)
proton_label.grid(row=1, column=0), proton_entry.grid(row=1, column=1)

# Function to be run in a separate thread for model training
def train_model_thread():
    global model, X_train, y_train, rtplot_ax
    real_time_plot = RealTimePlot(rtplot_ax)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[real_time_plot])

# Function to start the thread
def start_training_thread():
    training_thread = threading.Thread(target=train_model_thread)
    training_thread.start()

train_button = ttk.Button(root, text='Train Model', command=start_training_thread)
train_button.grid(row=5, columnspan=2)

# Add an "Update Plot" Button
update_button = ttk.Button(root, text='Update Plot', command=update_3D_plot)
update_button.grid(row=4, columnspan=2)

predict_button = ttk.Button(root, text='Predict', command=predict_binding_energy)
predict_button.grid(row=2, columnspan=2)

result_label = ttk.Label(root, text='')
result_label.grid(row=3, columnspan=2)

# Add Matplotlib Figure for real-time plotting
rtplot_fig, rtplot_ax = plt.subplots(figsize=(4, 4))
canvas_rtplot = FigureCanvasTkAgg(rtplot_fig, master=root)
canvas_rtplot_widget = canvas_rtplot.get_tk_widget()
canvas_rtplot_widget.grid(row=6, column=0)  # Changed column to 0

# Create a Matplotlib Figure for 3D plotting
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(4, 4))

# Add Matplotlib Figure for 3D plotting to Tkinter
canvas_3dplot = FigureCanvasTkAgg(fig, master=root)
canvas_3dplot_widget = canvas_3dplot.get_tk_widget()
canvas_3dplot_widget.grid(row=6, column=1)

# Create a frame to hold the toolbar for real-time plotting
toolbar_frame = tk.Frame(master=root)
toolbar_frame.grid(row=7, column=0)

# Create a frame to hold the toolbar for 3D plotting
toolbar_frame_3d = tk.Frame(master=root)
toolbar_frame_3d.grid(row=7, column=1)

# Add toolbar for the real-time plot inside the frame
toolbar = NavigationToolbar2Tk(canvas_rtplot, toolbar_frame)
toolbar.pack()
toolbar_frame.grid(row=7, column=0)

# Add toolbar for the 3D plot inside the frame
toolbar_3d = NavigationToolbar2Tk(canvas_3dplot, toolbar_frame_3d)
toolbar_3d.pack()
toolbar_frame_3d.grid(row=7, column=1)

# Initialize model but do not train yet
model = Sequential([Dense(16, input_dim=2, activation='relu'), Dense(16, activation='relu'), 
                    Dense(8, activation='relu'), Dropout(0.2), Dense(1, activation='linear')]) 
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) 

root.mainloop()