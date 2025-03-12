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

The implementation includes both deterministic and stochastic versions of the SEIR model, as well as SIR (Susceptible-Infected-Recovered) variants.

### 2. Change Point Detection (`choice_start_day.py`)

Methods for determining the optimal day to switch from observed data to model prediction:
- `cpoint_norm_var`: Uses Bayesian Information Criterion for normal variance changepoint detection
- `cpoint_roll_var`: Detects change based on rolling variance crossing a threshold
- `cpoint_roll_var_seq`: Identifies consecutive days of low variance 
- `cpoint_roll_var_npeople`: Waits for a minimum infected population before looking for variance change

### 3. Beta and Infected Prediction (`predict_Beta_I.py`)

Multiple methods for predicting the infection rate parameter (Beta) and corresponding Infected populations:
- Simple methods: last value, rolling mean, expanding mean
- Statistical methods: biexponential decay fitting
- Machine learning methods: regression and LSTM models

## Machine Learning Models

The project includes several pre-trained models:
- `regression_day_for_seir.joblib`: Regression model using day as the only feature
- `regression_day_SEIR_prev_I_for_seir.joblib`: Enhanced regression model using day, SEIR compartment values, and previous infection data
- `lstm_day_E_prev_I_for_seir.keras`: LSTM model for time series prediction based on day, exposed population, and previous infection rates

## Usage

1. Prepare seed data in a pandas DataFrame with columns for S, E, I, R, and Beta values
2. Choose a start day method using `choice_start_day.choose_method()`
3. Select a Beta prediction method and predict future Beta values with `predict_beta()`
4. The functions will return predicted values for both Beta and the Infected population

## Example

```python
import pandas as pd
import numpy as np
from choice_start_day import choose_method
from predict_Beta_I import predict_beta

# Load seed data
seed_df = pd.read_csv('path/to/seed_data.csv')

# Choose start day for prediction
start_day = choose_method(seed_df, 'roll_var')

# Predict for the next 30 days
predicted_days = np.arange(start_day, start_day + 30)

# Set SEIR parameters
sigma = 0.2  # Rate of progression from Exposed to Infected
gamma = 0.1  # Recovery rate

# Make predictions using regression model
beggining_beta, predicted_beta, predicted_I = predict_beta(
    'seir',                         # Mathematical model
    seed_df,                        # Seed data
    'regression (day)',             # Beta prediction method
    predicted_days,                 # Days to predict
    True,                           # Use stochastic model
    5,                              # Number of stochastic trajectories
    sigma,                          # Sigma parameter
    gamma                           # Gamma parameter
)

# Results in predicted_beta and predicted_I
```

## Model Training

The project includes Jupyter notebooks for training regression models:
- `save_model_regression_day_for_seir.ipynb`: Trains a simple day-based model
- `save_model_regression_day_SEIR_prev_I_for_seir.ipynb`: Trains an advanced model with additional features

## Dependencies

- NumPy
- pandas
- scikit-learn
- TensorFlow/Keras
- SciPy
- joblib

## Technical Notes

- The Beta parameter represents the infection rate in the SEIR model
- Stochastic models use binomial sampling to introduce randomness into predictions
- ML models are trained on normalized log-transformed Beta values to ensure stable predictions
- LSTM models use a sliding window approach to maintain temporal context
