# Masters_project - Machine learning for nuclear astrophysics

## Project Description
This project aims to use machine learning to analyze nuclear data, with a specific focus on applications in astrophysics. It uses neural networks (NNs) to predict and bound astrophysical properties such as the crustal composition of neutron stars, rapid neutron capture in supernovae, and elemental composition changes in stars over their lifecycle.

## Folder Structure (May be incomplete)

- **Test NN**: This directory contains the initial neural network implementation.
  - `AME2020_1.txt` - Full Atomic Mass Evaluation (AME) 2020 part 1 dataset.
  - `AME2020_converted.csv` - Processed dataset for B/A values.
  - `DataProcessing.py` - Python script for converting AME text data to a CSV format.
  - `Simple_NN.py` - Python script defining and training the neural network.
  - `trained_model.h5` - Saved neural network model for future use.

- **Bayesian NN**: This directory features the development of a Bayesian neural network.
  - `BayesianNN_V1_fail.py` - Initial attempt to integrate a BNN with a prior GUI.
  - `BayesianNN_V2_simple.py` - Successful implementation of a BNN with GUI elements removed for simplicity.

## Installation

To run the scripts in this repository, clone it to your local machine: git clone https://github.com/thelimelines/Masters_project

Ensure you have the necessary dependencies installed, including Python and relevant libraries such as TensorFlow, Pandas, etc.

## Usage

So far, each script in the repository is standalone. Run `DataProcessing.py` to convert the AME2020 data from text to CSV format. The `Simple_NN.py` and `BayesianNN_V2_simple.py` scripts can be run to train the respective neural networks.

## Contributing

Contributions to this project are not welcome at this time as I work on my master's degree. I intend to publicise this fully at the end of the year. Until then, feel free to branch the repo, citing me appropriately.

## License

[YET TO BE OFFICIALLY ASSIGNED] All work is open under attribution

## Contact

For further information or queries, please contact bjr539@york.ac.uk
