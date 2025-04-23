# SEIR Model Epidemic Prediction System

## Overview

This project implements a system for predicting epidemic dynamics using SEIR (Susceptible-Exposed-Infected-Recovered) models. The code provides various methods for determining infection rate parameters (Beta) and predicting the trajectory of an epidemic based on early observed data.

## Key Components

### 1. SEIR Model Implementation (`seir_discrete.py`)

The core epidemiological model that simulates disease spread through a population with four compartments:
- **S**: Susceptible individuals
- **E**: Exposed individuals (infected but not yet infectious)
- **I**: Infectious individuals
- **R**: Recovered individuals

The implementation includes both deterministic and stochastic versions of the SEIR model, as well as SIR variants.

### 2. Change Point Detection (`choice_start_day.py`)

Methods for determining the optimal day to switch from observed data to model prediction:
- `cpoint_roll_var`: Detects change based on rolling variance crossing a threshold
- `cpoint_roll_var_seq`: Identifies consecutive days of low variance 
- `cpoint_roll_var_npeople`: Waits for a minimum infected population before looking for variance change

### 3. Beta and Infected Prediction (`predict_Beta_I.py`)

Multiple methods for predicting the infection rate parameter (Beta) and corresponding Infected populations:
- Simple methods: last value, rolling mean, expanding mean
- Statistical methods: biexponential decay fitting
- ML methods: regression and LSTM models

### 4. Main Program (`plot_methods.ipynb`)

The main function that orchestrates the epidemic prediction process:

```python

def main_f(I_prediction_method, stochastic, count_stoch_line, 
           beta_prediction_method, type_start_day, seed_numbers,
           show_fig_flag, ax = None, 
           features_reg = ['day','prev_I','S','E','I','R'])
```

#### Parameters:
- **I_prediction_method**: Mathematical model for constructing the trajectory of infected individuals ['seir']
- **stochastic**: Boolean indicator for including stochastic trajectories
- **count_stoch_line**: Number of stochastic trajectories to generate
- **beta_prediction_method**: Method for predicting Beta values, including:
  - 'last_value'
  - 'rolling mean last value'
  - 'expanding mean last value'
  - 'biexponential decay'
  - 'median beta'
  - 'regression (day)'
  - 'median beta;\nshifted forecast'
  - 'regression (day);\nshifted forecast'
  - 'regression (day);\nincremental learning'
  - 'regression (day, SEIR, previous I)'
  - 'lstm (day, E, previous I)'
- **type_start_day**: Method for choosing the switch day from observed to predicted data:
  - 'roll_var'
  - 'roll_var_seq'
  - 'roll_var_npeople'
  - Or a specific integer day (20, 30, 40, etc.)
- **seed_numbers**: List of seed numbers for the experiments
- **show_fig_flag**: Boolean flag to display plots
- **save_fig_flag**: Boolean flag to save plots

#### Outputs:
- Single seed: Returns prediction data for detailed analysis
- Multiple seeds: Returns metrics lists (RMSE for I and Beta, peak information) and generates visualization plots

#### Visualization:
The program includes comprehensive plotting capabilities through the `plot_one` function:
- Displays actual vs. predicted infected curves
- Shows actual vs. predicted Beta parameter values
- Marks the switch point between observed and predicted data
- Includes confidence intervals for stochastic predictions
- Provides detailed metrics: peak timing, peak magnitude, RMSE, and execution time

## Machine Learning Models

The project includes several pre-trained models:
- `regression_day_for_seir.joblib`: Day-based regression model
- `regression_day_SEIR_prev_I_for_seir.joblib`: Enhanced regression with SEIR components
- `lstm_day_E_prev_I_for_seir.keras`: LSTM model for time series prediction

## Training Data

The training data is generated from regular network models of epidemic spread. These simulations use:
- Various network topologies to represent different population contact patterns
- Stochastic processes to introduce realistic variability
- Multiple epidemic seeds to capture different outbreak scenarios

## Model Training

The project includes three Jupyter notebooks for training ML models:

### 1. `save_model_regression_day_for_seir.ipynb`
Trains a polynomial regression using day as the sole feature. Processes multiple seed files and transforms Beta values logarithmically.

### 2. `save_model_regression_day_SEIR_prev_I_for_seir.ipynb`
Enhanced regression model using day, SEIR compartment values, and previous infection data as features.

### 3. `save_model_lstm_day_E_prev_I_for_seir.ipynb`
LSTM network for time series prediction with sliding window approach. Uses two LSTM layers with dropout for regularization.

## Dependencies

- NumPy
- pandas
- scikit-learn
- TensorFlow/Keras
- SciPy
- joblib
- matplotlib
- time
- math
- warnings

## Technical Notes

- Beta parameter represents infection rate in the SEIR model
- Stochastic models use binomial sampling for randomness
- ML models trained on normalized log-transformed Beta values
- LSTM models maintain temporal context via sliding windows
- Visualization uses matplotlib's dual-axis plotting for combined display of infected counts and Beta values
