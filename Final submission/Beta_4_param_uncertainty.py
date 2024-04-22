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


# Define variational posterior weight distribution as multivariate Gaussian.
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

hidden_units = [16, 8]
learning_rate = 0.001

def train_network(model, loss, train_dataset, test_dataset, num_epochs):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

def create_model_inputs():
    inputs = {}
    for feature_name in features:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

def create_probablistic_bnn_model(train_size):
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
            activation="sigmoid",
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

num_epochs = 2000
prob_bnn_model = create_probablistic_bnn_model(train_size)
train_network(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset,num_epochs)

sample = 20
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[
    0
]

prediction_distribution = prob_bnn_model(examples)
prediction_mean = prediction_distribution.mean().numpy().tolist()
prediction_stdv = prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
prediction_stdv = prediction_stdv.tolist()

for idx in range(sample):
    print(
        f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
        f"stddev: {round(prediction_stdv[idx][0], 2)}, "
        f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
        f" - Actual: {targets[idx]}"
    )