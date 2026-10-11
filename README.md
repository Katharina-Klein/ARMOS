# ARMOS

This repository contains a selected subset of the code developed for the experiments reported in the accompanying paper *Autoregressive Model Output Statistics*. The included scripts cover data preprocessing and visualization, the main model implementation, and the experimental setup. Some additional code used for generating result figures is not included. The repository is intended to document the implementation and support understanding and use of the model. The original experimental environment is no longer available, and compatibility with current Python and TensorFlow/Keras versions has not been verified.


## Repository structure

The repository is organized as follows:

- `ARMOS/`: Functions implementing the ARMOS model.
- `DATA/`: Scripts for data extraction, preprocessing, analysis, and visualization.
- `EMOS/`: Functions implementing the EMOS model and Schaake Shuffle algorithm.
- `EXPERIMENTS/`: Scripts used to run the experiments reported in the paper.
- `RESULTS/`: Functions for calculating scores and producing result plots.

The main experiment scripts are located in `EXPERIMENTS/`. These scripts import functions from the other directories.


## File overview

### `ARMOS/`

- `armos_model.py`: Implements the ARMOS model for different loss functions.

### `DATA/`

- `data_extraction_forecasts.py` and `data_extraction_observations.py`: Extract the forecast and observation data used in the experiments from the raw files.
- `data_processing_errors.py` and `data_processing_normalize_full.py`: Process the data and prepare it for the subsequent analyses and model experiments.
- `data_loading.py`: Provides functions for loading the prepared data into a dataset of the format required by the experiments.
- The remaining files create the plots used in the initial exploratory analysis of the data.

### `EMOS/`

- `emos_model.py`: Implements the EMOS model.
- `copula.py`: Implements the Schaake Shuffle algorithm.

### `EXPERIMENTS/`

- `parameter_selection_linear.py`: Performs parameter selection for choosing the grid size used in the spatial variance computation, as well as the predictors used in the model.
- `emos_tune_global.py`, `armos_LS_tune_global.py`, `armos_MSE_tune_global.py` and `armos_ES_tune_global.py`: Perform global parameter tuning for EMOS and the different ARMOS variants.
- `emos_fit_final.py`, `armos_LS_fit_final.py`, `armos_MSE_fit_final.py` and `armos_ES_fit_final.py`: Fit the final EMOS and ARMOS models using the selected parameters.

### `RESULTS/`

- `scores.py`: Implements functions used to compute the verification scores.
- `plots.py`: Implements functions used to generate the result figures.

