import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit


# our functions
import seir_discrete 

import warnings
warnings.filterwarnings(action='ignore')


def load_saved_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def biexponential_decay_func(x,a,b,c): 
            return a*(np.exp(-b*x)- np.exp(-c*x))

def inc_learning(seed_df, start_day,model_path):
            model_il = load_saved_model(model_path)

            x2 = np.arange(start_day).reshape(-1, 1)
            y2 = seed_df.iloc[:start_day]['Beta'].replace(to_replace=0, method='ffill').values.reshape(-1, 1).ravel()
            y2 = np.log(y2)

            t = model_il.named_steps['standardscaler'].transform(x2)
            name_2nd = list(model_il.named_steps.keys())[1]
            t = model_il.named_steps[name_2nd].transform(t)

            if model_il.named_steps['sgdregressor'].warm_start:
                # for warm_start=True .use fit()
                model_il.named_steps['sgdregressor'].fit(t,y2)
            else:
                for i in range(3):
                    # for warm_start=False use .partial_fit()
                    model_il.named_steps['sgdregressor'].partial_fit(t,y2)
            
            return model_il

class LSTMPredictor:
    """
    Wraps the trained LSTM model to predict beta on a rolling window of
    [day, E, prev_I] (3 features). 
    The model was trained to predict normalized log_beta, so this class
    denormalizes the prediction and returns beta.
    """
    def __init__(self, model, full_scaler, window_size):
        self.model = model
        # Create a scaler for input features
        # Corrected feature_indices calculation:
        feature_indices = list(range(3))
        self.input_scaler = StandardScaler()
        self.input_scaler.mean_ = full_scaler.mean_[feature_indices]
        self.input_scaler.scale_ = full_scaler.scale_[feature_indices]
        self.input_scaler.var_ = full_scaler.var_[feature_indices]
        self.input_scaler.n_features_in_ = 3
        self.window_size = window_size
        self.buffer = []
        # Store target parameters for log_beta (7th column)
        self.target_mean = full_scaler.mean_[-1]
        self.target_scale = full_scaler.scale_[-1]
        
    def update_buffer(self, new_data):
        # new_data should be a list with 3 elements: [day, E, prev_I]
        self.buffer.append(new_data)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
    def predict_next(self):
        # Ensure the buffer has window_size rows
        if len(self.buffer) < self.window_size:
            padded = np.zeros((self.window_size, 3))
            padded[-len(self.buffer):] = self.buffer
        else:
            padded = np.array(self.buffer[-self.window_size:])
            
        scaled = self.input_scaler.transform(padded)
        scaled_window = scaled.reshape(1, self.window_size, 3)
        normalized_pred = self.model.predict(scaled_window, verbose=0)[0][0]
        # Denormalize to obtain the raw log_beta
        raw_log_beta = normalized_pred * self.target_scale + self.target_mean
        # Compute beta by exponentiating the log_beta
        predicted_beta = np.exp(raw_log_beta)
        return predicted_beta

def predict_beta(I_prediction_method, seed_df, beta_prediction_method, predicted_days, 
                 stochastic, count_stoch_line, sigma, gamma):
    '''
    Predict Beta values.

    Parameters:

    - I_prediction_method -- mathematical model for predicting Infected trajectories
        ['seir']
    - seed_df -- DataFrame of seed, created by a regular network
    - beta_prediction_method -- method for predicting Beta values
        ['last_value',
        'rolling mean last value',
        'expanding mean last value',
        'biexponential decay', 
        'median beta',
        'regression (day)'

        'median beta;\nshifted forecast',
        'regression (day);\nshifted forecast',
        'regression (day);\nincremental learning',
        'regression (day, SEIR, previous I)',       
        'lstm (day, E, previous I)']
    - predicted_days -- days for prediction
    - stochastic -- indicator of the presence of predicted trajectories by a stochastic mathematical model
    - count_stoch_line -- number of trajectories predicted by the stochastic mathematical model
    - sigma -- parameter of the SEIR-type mathematical model
    - gamma -- parameter of the SEIR-type mathematical model
    '''
    predicted_I = np.zeros((count_stoch_line+1, predicted_days.shape[0]))
    beggining_beta = []

    if beta_prediction_method == 'last value':
        predicted_beta = [seed_df.iloc[predicted_days[0]]['Beta'] 
                          for i in range(predicted_days.shape[0])]

    elif beta_prediction_method == 'rolling mean last value':
        window_size = 7
        beggining_beta = seed_df.iloc[:predicted_days[0]]['Beta'
                                                         ].rolling(window=window_size).mean()
        predicted_beta = [beggining_beta.iloc[-1] 
                          for i in range(predicted_days.shape[0])]
    
    elif beta_prediction_method == 'expanding mean last value':
        betas = seed_df.iloc[:predicted_days[0]]['Beta'].mean()
        predicted_beta = [betas for i in range(predicted_days.shape[0])]

    elif beta_prediction_method == 'expanding mean':
        betas = seed_df.iloc[:]['Beta'].expanding(1).mean().values
        beggining_beta = betas[:predicted_days[0]]
        predicted_beta = betas[predicted_days[0]:]

    elif beta_prediction_method == 'biexponential decay':
        given_betas = seed_df.iloc[:predicted_days[0]]['Beta'].values
        given_days = np.arange(predicted_days[0])
        coeffs, _ = curve_fit(biexponential_decay_func, given_days, given_betas)
        beggining_beta = biexponential_decay_func(given_days, *coeffs)
        predicted_beta = biexponential_decay_func(predicted_days, *coeffs)
        predicted_beta[predicted_beta < 0] = 0
        
    elif beta_prediction_method == 'median beta':
        betas = pd.read_csv('train/median_beta.csv')
        beggining_beta = betas.iloc[:predicted_days[0]]['median_beta'].values
        predicted_beta = betas.iloc[predicted_days[0]:]['median_beta'].values

    elif beta_prediction_method == 'median beta;\nshifted forecast':
        betas = pd.read_csv('train/median_beta.csv')
        beggining_beta = betas.iloc[:predicted_days[0]]['median_beta'].values
        predicted_beta = betas.iloc[predicted_days[0]:]['median_beta'].values
        change = seed_df['Beta'].rolling(min(predicted_days[0]-2,14)).mean()[predicted_days[0]]
        change = np.sign(change - predicted_beta[0]) * (np.abs(change - predicted_beta[0]))
        beggining_beta += change
        predicted_beta += change

    elif beta_prediction_method == 'regression (day)':
        model_path = 'regression_day_for_seir.joblib'
        model = load_saved_model(model_path)
        x_test = np.arange(0,predicted_days[0]).reshape(-1, 1)
        beggining_beta = np.exp(model.predict(x_test))
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))

    elif beta_prediction_method == 'regression (day);\nshifted forecast':
        model_path = 'regression_day_for_seir.joblib'
        model = load_saved_model(model_path)
        x_test = np.arange(0,predicted_days[0]).reshape(-1, 1)
        beggining_beta = np.exp(model.predict(x_test))
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))
        change = seed_df['Beta'].rolling(min(predicted_days[0]-2,14)).mean().iloc[predicted_days[0]]
        change = np.sign(change - predicted_beta[0]) * (np.abs(change - predicted_beta[0]))
        beggining_beta += change
        predicted_beta += change 

    elif beta_prediction_method == 'regression (day);\nincremental learning':
        model_path = 'regression_day_for_seir.joblib'
        model = inc_learning(seed_df, predicted_days[0], model_path)
        x_test = np.arange(0,predicted_days[0]).reshape(-1, 1)
        beggining_beta = np.exp(model.predict(x_test))
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))

    elif beta_prediction_method == 'regression (day, SEIR, previous I)':
        predicted_beta = np.empty((0,))
        S = np.zeros((count_stoch_line+1, 2))
        E = np.zeros((count_stoch_line+1, 2))
        R = np.zeros((count_stoch_line+1, 2))

        S[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['S']
        predicted_I[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['I']
        R[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['R']  
        E[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['E']     
        model_path = 'regression_day_SEIR_prev_I_for_seir.joblib'
        y = np.array([S[:,0], E[:,0], predicted_I[:,0], R[:,0]])
        y = y.T

        model = load_saved_model(model_path)
        prev_I = seed_df.iloc[predicted_days[0]-2:predicted_days[0]]['I'].to_numpy() if predicted_days[0] > 1 else np.array([0.0, 0.0])
        log_beta = model.predict([[predicted_days[0], S[0,0], E[0,0], predicted_I[0,0], R[0,0], prev_I[0]]])
        beta = np.exp(log_beta)[0]
        predicted_beta = np.append(predicted_beta,max(beta, 0))
        for idx in range(predicted_days.shape[0]-1):

            # prediction of the Infected compartment trajectory
            S[0,:], E[0,:], predicted_I[0,idx:idx+2], R[0,:] = predict_I(
                                          I_prediction_method, y[0], 
                                          predicted_days[idx:idx+2], 
                                          predicted_beta[idx], sigma, gamma, 
                                          'det', beta_t=False)   
            if stochastic:
                for i in range(count_stoch_line):
                    S[i+1,:], E[i+1,:], predicted_I[i+1,idx:idx+2], R[i+1,:] = predict_I(
                                                  I_prediction_method, y[i+1], 
                                                  predicted_days[idx:idx+2], 
                                                  predicted_beta[idx], sigma, gamma, 
                                                  'stoch', beta_t=False) 
           
            y = np.array([S[:,1], E[:,1], predicted_I[:,idx+1], R[:,1]])
            y = y.T
            if (idx == 0) or (idx == 1):
                log_beta = model.predict([[predicted_days[idx+1], S[0,1], E[0,1], predicted_I[0,idx+1], R[0,1], prev_I[idx]]])
            else:
                log_beta = model.predict([[predicted_days[idx+1], S[0,1], E[0,1], predicted_I[0,idx+1], R[0,1], predicted_I[0,idx-1]]])
            
            beta = np.exp(log_beta)[0]

            predicted_beta = np.append(predicted_beta, max(beta, 0))

    elif beta_prediction_method == 'lstm (day, E, previous I)':
        model_path = 'lstm_day_E_prev_I_for_seir.keras'
        full_scaler = joblib.load('lstm_day_E_prev_I_for_seir.pkl')
        model = load_model(model_path)
        predictor = LSTMPredictor(model, full_scaler, window_size=14)
        prev_I = seed_df.iloc[predicted_days[0]-2:predicted_days[0]]['I'].to_numpy() if predicted_days[0] > 1 else np.array([0.0, 0.0])
        seed_df['day'] = range(len(seed_df))
        seed_df['prev_I'] = seed_df['I'].shift(-2).fillna(0)
        predicted_beta = np.empty((0,))
        S = np.zeros((count_stoch_line+1, 2))
        E = np.zeros((count_stoch_line+1, 2))
        R = np.zeros((count_stoch_line+1, 2))

        S[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['S']
        predicted_I[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['I']
        R[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['R']  
        E[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['E']  

        # Initialize predictor buffer using the last 'window_size' days
        for i in range(max(0, predicted_days[0] - predictor.window_size + 1), predicted_days[0] + 1):
            row = seed_df.iloc[i]
            raw_features = [row['day'], row['E'], row['prev_I']]
            predictor.update_buffer(raw_features)
        y = np.array([S[:,0], E[:,0], predicted_I[:,0], R[:,0]])
        y = y.T
        
        for idx in range(predicted_days.shape[0]):
            predicted_beta = np.append(predicted_beta, predictor.predict_next())     
            if idx == predicted_days.shape[0]-1:
                break      
            # prediction of the Infected compartment trajectory
            S[0,:], E[0,:], predicted_I[0,idx:idx+2], R[0,:] = predict_I(I_prediction_method, y[0], 
                                    predicted_days[idx:idx+2], 
                                    predicted_beta[idx], sigma, gamma, 'det', beta_t=False)   
            if stochastic:
                for i in range(count_stoch_line):
                    S[i+1,:], E[i+1,:], predicted_I[i+1,idx:idx+2], R[i+1,:] = predict_I(
                                                  I_prediction_method, y[i+1], 
                                                  predicted_days[idx:idx+2], 
                                                  predicted_beta[idx], sigma, gamma, 
                                                  'stoch', beta_t=False) 
            y = np.array([S[:,1], E[:,1], predicted_I[:,idx+1], R[:,1]])
            y = y.T
            if idx == 0:
                predictor.update_buffer([predicted_days[idx+1], E[0,1], prev_I[1]])
            else:
                predictor.update_buffer([predicted_days[idx+1], E[0,1], predicted_I[0,idx-1]])
    
    return np.array(beggining_beta), np.array(predicted_beta), predicted_I 

def predict_I(I_prediction_method, y, 
              predicted_days, 
              predicted_beta, sigma, gamma, stype, beta_t=True):
    '''
    Predict Infected values.

    Parameters:

    - I_prediction_method -- mathematical model for predicting the Infected trajectory
        ['seir']
    - y -- compartment values on the day of switching to the mathematical model
    - predicted_days -- days for prediction
    - predicted_beta -- predicted Beta values
    - sigma -- parameter of the SEIR-type mathematical model
    - gamma -- parameter of the SEIR-type mathematical model
    - stype -- type of mathematical model
        ['stoch', 'det']
    '''
    S,E,I,R = seir_discrete.seir_model(y, predicted_days, 
                        predicted_beta, sigma, gamma, 
                        stype, beta_t).T

    return S,E,I,R