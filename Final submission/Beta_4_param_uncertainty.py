import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load the dataset and perform feature engineering
df = pd.read_csv('Beta_Half_Lives_4_parameter/Training_data.csv')
df['Pairing Term'] = np.where(df['Mass Number (A)'] % 2 == 0, 
                              np.where(df['Atomic Number (Z)'] % 2 == 0, 1, -1), 0)

# Define constants and selection
q_beta_choice = 'Q_beta- WS4 (keV)'
features = ['Mass Number (A)', 'Atomic Number (Z)', 'Pairing Term', q_beta_choice]
target = 'Beta Partial Half-Life (log(s))'
cleaned_df = df[features + [target]].dropna()

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df[features], cleaned_df[target], test_size=0.15, random_state=42
)

# TensorFlow dataset creation
batch_size = 64
num_epochs = 5000

# Convert pandas DataFrame to TensorFlow Dataset
def create_tf_dataset(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    return dataset.batch(batch_size)

train_dataset = create_tf_dataset(X_train, y_train)
test_dataset = create_tf_dataset(X_test, y_test)

# Bayesian Neural Network setup
def build_bnn_model(train_size, features, hidden_units=[32,8], learning_rate=0.001):
    inputs = {feature: layers.Input(shape=(1,), name=feature, dtype=tf.float32) for feature in features}
    x = layers.Concatenate()(list(inputs.values()))
    x = layers.BatchNormalization()(x)

    for units in hidden_units:
        x = tfp.layers.DenseVariational(units, make_prior_fn=prior, make_posterior_fn=posterior,
                                        kl_weight=1/(100*train_size), activation='ReLU')(x)
    
    distribution_params = layers.Dense(2)(x)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate),
                  loss=negative_loglikelihood,
                  metrics=[keras.metrics.RootMeanSquaredError()])
    return model

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential([
        tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfp.layers.MultivariateNormalTriL(n)
    ])

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

# Train and Evaluate the Model
def train_and_evaluate(model, train_dataset, test_dataset, num_epochs):
    # Fit the model and capture the history
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    # Evaluate the model separately (optional if you just need the plot)
    train_loss, train_rmse = model.evaluate(train_dataset, verbose=1)
    test_loss, test_rmse = model.evaluate(test_dataset, verbose=1)
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    return history

# Main execution
train_size = len(X_train)
model = build_bnn_model(train_size, features)
history = train_and_evaluate(model, train_dataset, test_dataset, num_epochs)

def make_predictions(model, dataset, sample_size=20):
    examples, targets = list(dataset.unbatch().shuffle(sample_size * 10).batch(sample_size))[0]
    prediction_distribution = model(examples)
    prediction_mean = prediction_distribution.mean().numpy()
    prediction_stddev = prediction_distribution.stddev().numpy()

    upper = prediction_mean + 1.96 * prediction_stddev
    lower = prediction_mean - 1.96 * prediction_stddev

    for idx in range(sample_size):
        print(
            f"Prediction mean: {prediction_mean[idx][0]:.2f}, "
            f"stddev: {prediction_stddev[idx][0]:.2f}, "
            f"95% CI: [{upper[idx][0]:.2f} - {lower[idx][0]:.2f}] "
            f" - Actual: {targets[idx]}"
        )

def prepare_and_predict(model, df, features, q_beta_choice, output_path='Final submission/Predicted_Beta_Half_Lives.csv'):
    filtered_df = df[df[q_beta_choice] > 0].dropna(subset=features)
    predict_dataset = tf.data.Dataset.from_tensor_slices(dict(filtered_df[features])).batch(64)

    predicted_means = []
    predicted_stddevs = []

    for batch in predict_dataset:
        distribution = model(batch)
        predicted_means.extend(distribution.mean().numpy().flatten())
        predicted_stddevs.extend(distribution.stddev().numpy().flatten())

    filtered_df['Predicted Half-Life (log(s))'] = predicted_means
    filtered_df['Uncertainty (std dev)'] = predicted_stddevs
    filtered_df['Lower CI (95%)'] = predicted_means - 1.96 * np.array(predicted_stddevs)
    filtered_df['Upper CI (95%)'] = predicted_means + 1.96 * np.array(predicted_stddevs)

    filtered_df.to_csv(output_path, index=False)
    print(f"Predictions for nuclei with Q_beta > 0 have been saved to '{output_path}'.")

# Example usage
make_predictions(model, test_dataset)
prepare_and_predict(model, df, features, q_beta_choice)

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 16})
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.legend()
plt.show()