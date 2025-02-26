import numpy as np


def seir_one_day(y, beta, sigma, gamma):
    
    S, E, I, R = y
    
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

   
    S = np.max([S + dSdt, 0])
    #S += dSdt
    E += dEdt
    I += dIdt
    R += dRdt
    
    return [S, E, I, R]


def seir_one_day_stoch(y, beta, sigma, gamma):
    
    S, E, I, R = y
    b_s_i = np.random.binomial(S, np.min([beta*I,1]))
    sigma_e = np.random.binomial(E, sigma)
    gamma_i = np.random.binomial(I, gamma)
    
    dSdt = -b_s_i
    dEdt = b_s_i - sigma_e
    dIdt = sigma_e - gamma_i
    dRdt = gamma_i
    
    #S = np.max([S + dSdt, 0])
    S += dSdt
    E += dEdt
    I += dIdt
    R += dRdt
    
    return [S, E, I, R]


def seir_model(y, predicted_days, beta, sigma, gamma, 
               stype='stoch', beta_t=False):
    res = np.zeros((predicted_days.shape[0], 4))
    res[0] = y
    beta_val = beta
    
    for time_stamp in predicted_days[:-1]-predicted_days[0]:
        if beta_t:
            beta_val = beta[time_stamp]
            
        if stype=='stoch':
            res[time_stamp+1] = seir_one_day_stoch(res[time_stamp], 
                                         beta_val, sigma, gamma)
        else:
            res[time_stamp+1] = seir_one_day(res[time_stamp], 
                                         beta_val, sigma, gamma)
        
    return res
    

def sir_model(y, predicted_days, beta, gamma, stype='stoch', beta_t=False):
    res = np.zeros((predicted_days.shape[0], 3))
    res[0] = y
    beta_val = beta
    
    for time_stamp in predicted_days[:-1]-predicted_days[0]:
        if beta_t:
            beta_val = beta[time_stamp]
        if stype=='stoch':
            res[time_stamp+1] = sir_one_day_stoch(res[time_stamp], 
                                         beta_val, gamma)
        else:
            res[time_stamp+1] = sir_one_day(res[time_stamp], 
                                         beta_val, gamma)
        
    return res


def sir_one_day(y, beta, gamma):
    
    S, I, R = y
    
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    
    S = np.max([S + dSdt, 0])
    I += dIdt
    R += dRdt
    
    return [S, I, R]


def sir_one_day_stoch(y, beta, gamma):
    
    S, I, R = y
    b_s_i = np.random.binomial(S, np.min([beta*I,1]))
    gamma_i = np.random.binomial(I, gamma)
    
    dSdt = -b_s_i
    dIdt = b_s_i - gamma_i
    dRdt = gamma_i
    
    #S = np.max([S + dSdt, 0])
    S += dSdt
    I += dIdt
    R += dRdt
    
    return [S, I, R]