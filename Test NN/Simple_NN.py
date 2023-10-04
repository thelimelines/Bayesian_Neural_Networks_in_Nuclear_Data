# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
import tkinter as tk
from tkinter import ttk

# Define custom callback for real-time plotting
class RealTimePlot(Callback):
    def on_train_begin(self, logs={}):
        self.losses, self.val_losses = [], []
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.ax.clear() # needed to remove old plot and maintain a single line of data
        self.ax.plot(self.losses, label='loss')
        self.ax.plot(self.val_losses, label='val_loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss Value')
        self.ax.legend()
        plt.pause(0.1) # gives the gui time to update

# GUI to allow user to test model on real data
def predict_binding_energy():
    neutron_number, proton_number = float(neutron_entry.get()), float(proton_entry.get())
    input_data = scaler_x.transform(np.array([[neutron_number, proton_number]])) # scales user input to model inputs
    prediction = scaler_y.inverse_transform(model.predict(input_data)) # scales model output to real output (keV)
    in_training_set = any(np.all(X_train == input_data, axis=1)) # checks if the user input was used in training
    actual_value = df.loc[(df['Neutron Number'] == neutron_number) & (df['Proton Number'] == proton_number)] # retrieves real value
    actual_value = actual_value['Binding Energy per Nucleon (keV)'].values[0] if not actual_value.empty else 'N/A' # string manipulation for UI
    result_label.config(text=f"In Training Set: {in_training_set}\nActual Value: {actual_value}\nPredicted Value: {prediction[0][0]:.2f}")

# Load and preprocess data
df = pd.read_csv('Test NN\\AME2020_converted.csv') # dataframe retrieved from trimmed AME

# X and Y are the inputs and outputs of the model respectively, note X is an array size 2
X, y = df[['Neutron Number', 'Proton Number']].values, df['Binding Energy per Nucleon (keV)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #30% testing, 70%training, random_State is a seed set for reproducability

scaler_x, scaler_y = StandardScaler().fit(X_train), StandardScaler().fit(y_train.reshape(-1, 1))
X_train, X_test = scaler_x.transform(X_train), scaler_x.transform(X_test)
y_train, y_test = scaler_y.transform(y_train.reshape(-1, 1)).flatten(), scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Initialize and train model
model = Sequential([Dense(16, input_dim=2, activation='relu'), Dense(16, activation='relu'), # 2-16-8-1 network map
                    Dense(8, activation='relu'), Dropout(0.2), Dense(1, activation='linear')]) # 'relu' is max(0,x), simplest way I could think to add non-linearity
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) # adam is the optimisation algorithm, mae is just mean absolute error
real_time_plot = RealTimePlot()
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[real_time_plot]) # Train the network and save the historic loss (plotted real time in this code)

# Evaluate and save model
test_loss, test_mae = model.evaluate(X_test, y_test)
model.save('Test NN\\trained_model.h5')

# GUI setup
root = tk.Tk()
root.title('Neural Network Prediction for Nuclear Binding Energy')

neutron_label, neutron_entry = ttk.Label(root, text='Neutron Number:'), ttk.Entry(root)
neutron_label.grid(row=0, column=0), neutron_entry.grid(row=0, column=1)

proton_label, proton_entry = ttk.Label(root, text='Proton Number:'), ttk.Entry(root)
proton_label.grid(row=1, column=0), proton_entry.grid(row=1, column=1)

predict_button = ttk.Button(root, text='Predict', command=predict_binding_energy)
predict_button.grid(row=2, columnspan=2)

result_label = ttk.Label(root, text='')
result_label.grid(row=3, columnspan=2)

root.mainloop()