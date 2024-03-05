import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_probability as tfp
import random

# Function to create posterior
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                                     scale=1 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# Function to create prior
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

# Load the dataset
df = pd.read_csv('Beta half life\\Beta_Half_Lives.csv')

# Prepare the data
X = df[['Mass Number', 'Atomic Number']].values
y = np.log(df['Beta Partial Half-Life (s)'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Define the Bayesian Neural Network model
model = tf.keras.Sequential([
    tfp.layers.DenseVariational(50, input_dim=2, activation='relu',
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable),
    tfp.layers.DenseVariational(50, activation='relu',
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable),
    tfp.layers.DenseVariational(50, activation='relu',
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable),
    tfp.layers.DenseVariational(1, activation='linear',
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train_scaled, y_train, epochs=5000, batch_size=32)

# Predict with uncertainty (Monte Carlo Dropout)
def predict_with_uncertainty(input_data, n_iter=100):
    predictions = np.array([model(input_data, training=True) for _ in range(n_iter)])
    predictions = predictions.squeeze()
    
    mean_prediction = np.mean(predictions, axis=0)
    std_dev_prediction = np.std(predictions, axis=0)
    
    return mean_prediction, std_dev_prediction

# Test data scaled
X_test_scaled = scaler_X.transform(X_test)

# Prediction
mean_preds_log_scale, std_preds_log_scale = predict_with_uncertainty(X_test_scaled)

# Rescale the predictions back to original scale
mean_preds = np.exp(mean_preds_log_scale)
std_preds = np.exp(std_preds_log_scale)

# Identify mass and atomic number combinations not in the training set
train_set = set([tuple(x) for x in X_train])
all_set = set([tuple(x) for x in X])
diff_set = all_set - train_set

# Randomly select 50 of these combinations
random_samples = np.array(random.sample(list(diff_set), 50))

# Make predictions using the trained Bayesian Neural Network
random_samples_scaled = scaler_X.transform(random_samples)
mean_preds_log_scale, std_preds_log_scale = predict_with_uncertainty(random_samples_scaled)

# Convert predictions back from log scale to the original scale
mean_preds_rescaled = np.exp(mean_preds_log_scale)  # Adjusting for the log transformation
std_preds_rescaled = np.exp(std_preds_log_scale)  # Adjusting for the log transformation

# Retrieve the actual values and uncertainties from the dataset
actual_values = []
for sample in random_samples:
    actual_row = df.loc[(df['Mass Number'] == sample[0]) & (df['Atomic Number'] == sample[1])]
    actual_values.append(actual_row['Beta Partial Half-Life (s)'].values[0])

# Print predictions, uncertainties, and actual values
for i in range(50):
    print(f"Mass Number: {random_samples[i][0]}, Atomic Number: {random_samples[i][1]}")
    print(f"Prediction: {mean_preds_rescaled[i]:.2f} ± {std_preds_rescaled[i]:.2f} s")

    # Check if the actual value is within the prediction uncertainty
    within_uncertainty = mean_preds_rescaled[i] - std_preds_rescaled[i] <= actual_values[i] <= mean_preds_rescaled[i] + std_preds_rescaled[i]
    symbol = "(:" if within_uncertainty else "X"
    
    print(f"Actual Value: {actual_values[i]:.2f} s {symbol}")
    print("---")