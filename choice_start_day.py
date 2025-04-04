from sklearn.preprocessing import MinMaxScaler
import numpy as np


def choose_method(seed_df, start_day):
    if start_day == 'roll_var':
        start_day_v = cpoint_roll_var(seed_df)
    elif start_day == 'norm_var':
        start_day_v = cpoint_norm_var(seed_df)
    elif start_day == 'roll_var_seq':
        start_day_v = cpoint_roll_var_seq(seed_df)
    elif start_day == 'roll_var_npeople':
        start_day_v = cpoint_roll_var_npeople(seed_df)
    else:
        start_day_v = start_day
    
    return start_day_v


def cpoint_norm_var(seed_df):
    
    # http://www.claudiobellei.com/2016/11/15/changepoint-frequentist/ 
    # type=="normal-var"

    data = seed_df.Beta.values
    n = len(data)
    tau = np.arange(1,n)
    criterion = 1*np.log(n) #Bayesian Information Criterion
    #criterion = 2
  
    eps = 1.e-8 #to avoid zeros in denominator

    std0 = np.std(data)
    std1 = np.asarray([np.std(data[0:i]) for i in range(1,n)],dtype=float) + eps
    std2 = np.asarray([np.std(data[i:]) for i in range(1,n)],dtype=float) + eps
    R = n*np.log(std0) - tau*np.log(std1) - (n-tau)*np.log(std2)
    G  = np.max(R)
    cpoint = int(np.where(R==G)[0]) + 1

    teststat = 2*G

    if teststat > criterion:
        return cpoint
    else:
        return 30
        


# look for a change in variance (< 5%)
def cpoint_roll_var(seed_df, thresh = 0.05):
    scaler = MinMaxScaler()

    var_vals = seed_df.Beta.rolling(7).var()
    scaled_varv = scaler.fit_transform(var_vals.values.reshape(-1, 1))

    cpoint = np.nanmin(np.where(scaled_varv < thresh)[0])   

    if cpoint < 7:
        return 7
    else:
        return cpoint

# look for a change in variance (< 5%) which holds 2 days
def cpoint_roll_var_seq(seed_df, thresh = 0.05):
    scaler = MinMaxScaler()

    var_vals = seed_df.Beta.rolling(7).var()
    scaled_varv = scaler.fit_transform(var_vals.values.reshape(-1, 1))

    # where var <= threshold
    ids = np.where(scaled_varv < thresh)[0]
    # arrays of consequent values
    ids_splits = np.split(ids, np.where(np.diff(ids)!=1)[0]+1)
    split_shapes = np.array([i.shape[0] for i in ids_splits])
    
    # at least n consequent values
    split_id = np.where(split_shapes>=2)[0][0]
    # first value of the needed group
    cpoint = ids_splits[split_id][0]

    return cpoint


# wait until 1% of population is infected, 
# and only then look for a change in variance
def cpoint_roll_var_npeople(seed_df, thresh = 0.1, n_people=100):
    scaler = MinMaxScaler()

    var_vals = seed_df.Beta.rolling(7).var()
    scaled_varv = scaler.fit_transform(var_vals.values.reshape(-1, 1))
    
    day_with_npeople = seed_df[seed_df.I >= n_people].index[0]
    cpoint = np.nanmin(np.where(scaled_varv[day_with_npeople:] < thresh)[0])   
    if cpoint + day_with_npeople < 14:
        return 14
    else:
        return cpoint + day_with_npeople

