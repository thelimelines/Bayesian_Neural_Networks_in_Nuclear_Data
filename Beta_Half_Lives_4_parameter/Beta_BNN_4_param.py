import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Beta_Half_Lives_4_parameter\Training_data.csv')

# Feature Engineering for Pairing Term
df['Pairing Term'] = np.where(df['Mass Number (A)'] % 2 == 0, 
                              np.where(df['Atomic Number (Z)'] % 2 == 0, 1, -1), 
                              0)

# Select features and target
q_beta_choice = 'Q_beta- WS4 (keV)'  # or 'Q_beta- WS4+RBF (keV)'

features = ['Mass Number (A)', 'Atomic Number (Z)', 'Pairing Term', q_beta_choice]  # q_beta_choice is either 'Q_beta- WS4 (keV)' or 'Q_beta- WS4+RBF (keV)'
target = 'Beta Partial Half-Life (log(s))'

# Cleaning data: Remove rows with NaN in any specified feature or the target
cleaned_df = df[features + [target]].dropna()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df[features], cleaned_df[target], test_size=0.15, random_state=42
)

# Setting the train_size variable
train_size = len(X_train)

# Convert to TensorFlow datasets
batch_size = 64

# Convert to TensorFlow datasets
def create_tf_dataset(data, labels):
    dataset_dict = {feature: data[feature].values[:, None] for feature in data.columns}
    return tf.data.Dataset.from_tensor_slices((dataset_dict, labels)).batch(batch_size)

train_dataset = create_tf_dataset(X_train, y_train)
test_dataset = create_tf_dataset(X_test, y_test)

# Define prior guesses as multivariate normal with means=0, SDs=1 and no initial covariences
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Normal.
# Note that the learnable parameters for this distribution are the means, variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

hidden_units = [20, 10]
learning_rate = 0.001

def train_network(model, loss, train_dataset, test_dataset, num_epochs):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset, verbose=1)
    print("Model training finished.")

    # Plotting the training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.title('RMSE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

def create_model_inputs():
    inputs = {}
    for feature_name in features:
        inputs[feature_name] = layers.Input(name=feature_name, shape=(1,), dtype=tf.float32)
    return inputs

def create_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="relu",
        )(features)

    # The output is a single point estimate.
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

mse_loss = tf.keras.losses.MeanSquaredError()

num_epochs = 500 # Passes through full dataset
bnn_model_full = create_bnn_model(train_size)
train_network(bnn_model_full, mse_loss, train_dataset, test_dataset, num_epochs)

samples = 10 # number of predictions
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(samples))[0]

def compute_predictions(model, iterations=100): # For each prediction, run 'iterations' number of passes through the network
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist() 
    for idx in range(samples):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
            f"Actual: {targets[idx]}"
        )

compute_predictions(bnn_model_full)