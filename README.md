# Masters_project - Machine learning for nuclear astrophysics

## Project Description
This project aims to use statistics based machine learning to predict beta decay partial half-lives in order to restrict uncertainties in r-process nucleosynthesis/

## Folder Structure and overview

- **Bayesian NN**: This directory features the development of a Bayesian neural network.
  - `BayesianNN_V1_fail.py` - Deprecated: Initial attempt to integrate a Bayesian Neural Network (BNN) with a GUI.
  - `BayesianNN_V2_simple.py` - Successful implementation of a simplified BNN without GUI components.

- **Beta half life**: This directory contains scripts and data related to beta half-life experiments using Bayesian Neural Networks.
  - `Beta_BNN.py` - Deprecated: Bayesian Neural Network with 2 parameters, found to be unstable.
  - `Beta_BNN_V2.py` - Stable version of the 2-parameter Bayesian Neural Network.
  - `Beta_Half_lives.csv` - Processed data for beta partial half-lives.
  - `Beta_processing.py` - Script to convert Nubase data to partial half-lives.
  - `Log_Beta_Half_Lives.csv` - Table of isotopes with logarithmic half-lives.
  - `nubase_4.mas20.txt` - Raw Nubase2020 dataset.

- **Beta Half Lives 4 parameter**: This directory expands the model to 4 parameters for beta half-life prediction.
  - `Beta_BNN_4_param.py` - Deprecated: Original 4-parameter Bayesian Neural Network model using Keras.
  - `Beta_BNN_4_trimmed.py` - Revised 4-parameter BNN for improved readability and performance.
  - `Data_merge.py` - Script to merge datasets into a single training dataset.
  - `Log_Beta_Half_Lives.csv` - Table of isotopes with logarithmic half-lives.
  - `Q_Processed_WS4.csv` - Derived Q_beta values from WS4 data.
  - `Training_data.csv` - Compiled training data for neural network modeling.
  - `WS4_Q_calc.py` - Calculates Q_beta values from WS4 data.
  - `WS4_RBF.txt` - Raw WS4 dataset.

- **Finalised network and tools**: This directory includes the final neural network models and visualization tools.
  - `Beta_4_param_uncertainty.py` - Final model for 4-parameter Bayesian Neural Network with uncertainty.
  - `Elemental_plot_slider.py` - Visualization tool for predictions on atomic number (Z).
  - `Neutron_plot_slider.py` - Visualization tool for predictions on neutron number (N).
  - `Predicted_Beta_Half_Lives FULL.csv` - Complete set of untrimmed predictions.
  - `Predicted_Beta_Half_Lives TRIMMED.csv` - Trimmed predictions for half-lives less than 10^6 seconds.
  - `Predicted_Beta_Half_Lives.csv` - Temporary database from Bayesian Neural Network predictions.
  - `Predicted_vs_Actual_Beta_Half_Lives.py` - Visualization script to compare predicted versus experimental beta half-lives.
  - `Training_data.csv` - Compiled training data for neural network modeling.

- **Test NN**: This directory contains the initial neural network implementation.
  - `AME2020_1.txt` - Full Atomic Mass Evaluation (AME) 2020 part 1 dataset.
  - `AME2020_converted.csv` - Processed dataset for B/A values only.
  - `DataProcessing.py` - Python script for converting AME text data to a CSV format.
  - `Simple_NN.py` - Python script defining and training the neural network.
  - `trained_model.h5` - Saved neural network model for future use.

- **Visualisations**: This directory houses visualization tools for neural network models.
  - `Network vis.py` - Python script for visualizing the neural network architecture.
  - `NN.png` - Rendered image from the network visualization script.

- **Wine_BNN_test**: Tutorial example stripped from Keras documentation on Bayesian Neural Networks.
  - URL: [Keras Bayesian Neural Networks Tutorial](https://keras.io/examples/keras_recipes/bayesian_neural_networks/)

- **WS4 model**: This directory contains an attempt to reproduce WS4 model results.
  - `WS4.py` - Deprecated: Initial script trying to replicate WS4 results.

- `requirements.txt` - List of Python packages required for the project.


## Installation

**Requires python 3.11 to work**

To run the scripts in this repository, clone it to your local machine: git clone https://github.com/thelimelines/Masters_project

Install all dependencies at once using command: 'pip install -r requirements.txt' in the root folder's terminal.

## Usage

So far, each script in the repository is standalone. Run `DataProcessing.py` to convert the AME2020 data from text to CSV format. The `Simple_NN.py` and `BayesianNN_V2_simple.py` scripts can be run to train the respective neural networks and so on. CSVs are written on the fly and may be used by visualisation codes. Changing this pointer needs to be done manually within the scripts.

## Contributing

Contributions to this project are not welcome as this work is the product of a master's degree. However, feel free to branch and repurpose the repo, citing me appropriately.

## License

Copyright (c) 2024, Brodie Jean-Luke Rolph
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of York University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Contact

For further information or queries, please contact bjr539@york.ac.uk.