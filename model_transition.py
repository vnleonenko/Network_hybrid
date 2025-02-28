import seir_discrete 
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from scipy.optimize import curve_fit
import time
import math
import warnings
warnings.filterwarnings(action='ignore')


def plot_one(ax, 
             predicted_days, seed_df, predicted_I, predicted_beta,
             seed_number, execution_time):
    '''
    Построение графика для seed.
    
    Параметры:

    - ax -- область для графика
    - predicted_days -- дни предсказания
    - seed_df -- DataFrame of seed, созданный регулярной сетью
    - predicted_I -- предсказанные траектория компартмента Infected
    - predicted_beta -- предсказанные значения Beta
    - seed_number -- номер seed        
    - execution_time - время предсказания Beta   
    '''
    # подсчет RMSE для значений Infected и Beta
    actual_I = seed_df.iloc[predicted_days[0]:]['I'].values 
    rmse_I = rmse(actual_I, predicted_I[0])
    actual_Beta = seed_df.iloc[predicted_days[0]:]['Beta'].values 
    rmse_Beta = rmse(actual_Beta, predicted_beta)   

    # peak time and peak height
    actual_peak_height, pred_peak_height = seed_df['I'].max(), predicted_I[0].max()
    actual_peak_time, pred_peak_time = seed_df['I'].argmax(), \
                                        predicted_days[0]+predicted_I[0].argmax()
    rmse_ph = rmse([actual_peak_height], [pred_peak_height])
    rmse_pt = rmse([actual_peak_time], [pred_peak_time])
    
    # отображение границы перехода
    ax.axvline(predicted_days[0], color='red',ls=':')

    # отображение реальных и предсказанных значений Infected
    ax.plot(seed_df.index, seed_df.iloc[:]['I'].values , color='tab:blue', 
            label='Actual I')
    ax.plot(predicted_days, predicted_I[0],color='blue', ls='--', 
                alpha=0.9, label='Predicted I (det.)')
    
    # отображение траекторий стохастической мат. модели
    for i in range(predicted_I.shape[0]-1):
        if i==0:
            label='Predicted I (stoch.)'
        else:
            label=''
            
        ax.plot(predicted_days, predicted_I[i+1], color='tab:blue', ls='--', 
                alpha=0.3, label=label)
    
    # добавление названий осей
    ax.set_xlabel('Days')
    ax.set_ylabel('Infected', color='blue')
    ax.grid(True, alpha=0.3)
        
    ax_b = ax.twinx()
    # отображение реальных и предсказанных значений beta
    ax_b.plot(seed_df.index, seed_df['Beta'],  color='gray', ls='--', 
              alpha=0.4, label='Actual Beta')
    ax_b.plot(predicted_days, predicted_beta,color='green', ls='--', 
              alpha=0.7,label='Predicted Beta ')
    ax_b.set_ylabel("Beta", color='green')

    
    # добавление легенды и заголовков
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title(f'Seed {seed_number}, \n'+
                 f'RMSE I: {rmse_I:.2f}, RMSE Peak height: {rmse_ph:.2f}'+
                     f', RMSE Peak time: {rmse_pt} \n'+
                 f'RMSE beta: {rmse_Beta:.2e}, Predict time: {execution_time:.2e}',
                 fontsize=10)
    return rmse_I, rmse_Beta, \
            actual_peak_height, pred_peak_height, \
            actual_peak_time, pred_peak_time
    
    
def load_saved_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def predict_beta(I_prediction_method, seed_df, beta_prediction_method, predicted_days, 
                 stochastic, count_stoch_line, sigma, gamma):
    '''
    Предсказзание значений Beta.

    Параметры:

    - I_prediction_method -- математическая модель для предсказания траекторий Infected
        ['seir', 'sir']
    - seed_df -- DataFrame of seed, созданный регулярной сетью
    - beta_prediction_method -- метод предсказания значений Beta
        ['mean_const','expanding_mean','bi_exp_decay', 'polynom_of_day', 
        'polynom_of_day_prev_I','polynom_of_day_with_shift', 'polynom_SGDReg', 
        'polynom_SGDReg_with_rollingshift','polynom_SGDReg_with_future_training','lstm']
    - predicted_days -- дни предсказания
    - stochastic -- индикатор присутствия предсказанных  стохастической мат. моделью 
        траекторий Infected 
    - count_stoch_line -- количество предсказанных стохастической мат. моделью 
        траекторий Infected 
    - sigma -- параметр мат. модели типа SEIR
    - gamma -- параметр мат. модели типа SEIR и SIR
    '''
    predicted_I = np.zeros((count_stoch_line+1, predicted_days.shape[0]))
    execution_time = 0
    start_time = time.time()

    if beta_prediction_method == 'mean_const':
        betas = seed_df.iloc[:predicted_days[0]]['Beta'].mean()
        predicted_beta = [betas for i in range(predicted_days.shape[0])]

    elif beta_prediction_method == 'expanding_mean':
        betas = seed_df.iloc[:]['Beta'].expanding(1).mean().values
        predicted_beta = betas[predicted_days[0]:]

    elif beta_prediction_method == 'bi_exp_decay':
        def bi_exp_decay_func(x,a,b,c): 
            return a*(np.exp(-b*x)- np.exp(-c*x))
        given_betas = seed_df.iloc[:predicted_days[0]]['Beta'].values
        given_days = np.arange(predicted_days[0])
        coeffs, _ = curve_fit(bi_exp_decay_func, given_days, given_betas)
        predicted_beta = bi_exp_decay_func(predicted_days, *coeffs)
        predicted_beta[predicted_beta < 0] = 0
        
    elif beta_prediction_method == 'polynom_of_day':
        if I_prediction_method == 'seir':
            model_path = 'polynom_of_day_for_seir.joblib'
        else: 
            model_path = 'polynom_of_day_for_sir.joblib'
        # загрузка модели
        model = load_saved_model(model_path)
        # предсказываем значения Beta на оставшиеся дни
        log_beta = model.predict(predicted_days.reshape(-1,1))
        predicted_beta = np.exp(log_beta)

    elif beta_prediction_method == 'polynom_of_day_with_shift':
        if I_prediction_method == 'seir':
            model_path = 'polynom_of_day_for_seir.joblib'
        else: 
            model_path = 'polynom_of_day_for_sir.joblib'
        # загрузка модели
        model = load_saved_model(model_path)
        # предсказываем значения Beta на оставшиеся дни
        log_beta = model.predict(predicted_days.reshape(-1,1))
        predicted_beta = np.exp(log_beta)
        given_betas = seed_df.iloc[predicted_days[0]]['Beta']
        predicted_beta = predicted_beta  + np.sign(given_betas - predicted_beta[0]
                                                  ) * (np.abs(given_betas - predicted_beta[0]))

    elif beta_prediction_method == 'polynom_of_day_prev_I':
        predicted_beta = np.empty((0,))
        S = np.zeros((count_stoch_line+1, 2))
        E = np.zeros((count_stoch_line+1, 2))
        R = np.zeros((count_stoch_line+1, 2))
        # извлечение значений компартментов в день переключения на мат. модель
        S[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['S']
        predicted_I[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['I']
        R[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['R']  
        E[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['E']     
        if I_prediction_method == 'seir': 
            model_path = 'polynom_of_day_prev_I_for_seir.joblib'
        else: 
            model_path = 'polynom_of_day_prev_I_for_sir.joblib'
        
        y = np.array([S[:,0], E[:,0], predicted_I[:,0], R[:,0]])
        y = y.T
        # загрузка модели
        model = load_saved_model(model_path)
        prev_I = seed_df.iloc[predicted_days[0]-1]['I'] if predicted_days[0] > 0 else 0.0
        log_beta = model.predict([[predicted_days[0], prev_I]])
        beta = np.exp(log_beta)[0]
        predicted_beta = np.append(predicted_beta,max(beta, 0))
        for idx in range(predicted_days.shape[0]-1):
           
            # предсказание траектория компартмента Infected
            S[0,:], E[0,:], predicted_I[0,idx:idx+2], \
                R[0,:] = predict_I(I_prediction_method, y[0], 
                                   predicted_days[idx:idx+2], 
                                   predicted_beta[idx], sigma, gamma, 
                                   'det', beta_t=False)   
            if stochastic:
                for i in range(count_stoch_line):
                    S[i+1,:], E[i+1,:], predicted_I[i+1,idx:idx+2], \
                        R[i+1,:] = predict_I(I_prediction_method, y[i+1], 
                                             predicted_days[idx:idx+2], 
                                             predicted_beta[idx], sigma, gamma, 
                                             'stoch', beta_t=False) 
           
            y = np.array([S[:,1], E[:,1], predicted_I[:,idx+1], R[:,1]])
            y = y.T
            log_beta = model.predict([[predicted_days[idx+1], predicted_I[0,idx]]])
            beta = np.exp(log_beta)[0]
            predicted_beta = np.append(predicted_beta, max(beta, 0))
    
    elif beta_prediction_method == 'polynom_SGDReg':
        if I_prediction_method == 'seir': 
            model_path = 'polynom_SGDReg_for_seir.joblib'
        else: 
            model_path = 'polynom_SGDReg_for_sir.joblib'
        model = load_saved_model(model_path)
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))

    elif beta_prediction_method == 'polynom_SGDReg_with_rollingshift':
        if I_prediction_method == 'seir': 
            model_path = 'polynom_SGDReg_for_seir.joblib'
        else: 
            model_path = 'polynom_SGDReg_for_sir.joblib'
        model = load_saved_model(model_path)
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))
        change = seed_df['Beta'].rolling(14).mean().iloc[predicted_days[0]]
        #change = predicted_beta[0] - seed_df.iloc[:]['Beta'].rolling(14).mean()[predicted_days[0]]
        predicted_beta = predicted_beta  + np.sign(change - predicted_beta[0]
                                                  ) * (np.abs(change - predicted_beta[0]))

    elif beta_prediction_method == 'polynom_SGDReg_with_future_training':
        if I_prediction_method == 'seir': 
            model_path = 'polynom_SGDReg_for_seir.joblib'
        else: 
            model_path = 'polynom_SGDReg_for_sir.joblib'
        def inc_learning(seed_df, start_day,model_path):
            model_il = load_saved_model(model_path)

            x2 = np.arange(start_day).reshape(-1, 1)
            y2 = seed_df.iloc[:start_day]['Beta'].replace(to_replace=0, 
                                                          method='ffill'
                                                         ).values.reshape(-1, 1).ravel()
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
        model = inc_learning(seed_df, predicted_days[0], model_path)
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))
    
    elif beta_prediction_method == 'lstm':
        if I_prediction_method == 'seir': 
            model_path = 'lstm_for_seir.joblib'
        else: 
            model_path = 'lstm_for_sir.joblib'
        model = load_saved_model(model_path)

    elif beta_prediction_method == 'exp_decay':
        # заглушка
        predicted_beta = [0 for i in predicted_days]    
    
    elif beta_prediction_method == 'percentiles':
        # заглушка
        predicted_beta = [0 for i in predicted_days]
    
    end_time = time.time()
    execution_time += end_time - start_time
    return predicted_beta, execution_time, predicted_I 


def predict_I(I_prediction_method, y, 
              predicted_days, 
              predicted_beta, sigma, gamma, stype, beta_t=True):
    '''
    Предсказание значений Infected.

    Параметры:

    - I_prediction_method -- математическая модель для предсказания 
        траектории Infected
        ['seir', 'sir']
    - y -- значения компартментов в день переключения на мат. модель
    - predicted_days -- дни предсказания
    - predicted_beta -- предсказанные значения Beta
    - sigma -- параметр мат. модели типа SEIR
    - gamma -- параметр мат. модели типа SEIR и SIR
    - stype -- тип мат. модели
        ['stoch', 'det']
    '''
    
    if I_prediction_method == 'seir':
        S,E,I,R = seir_discrete.seir_model(y, predicted_days, 
                            predicted_beta, sigma, gamma, 
                            stype, beta_t).T
    else:
        if len(y) == 4:
            y = y[[0,2,3]]
        S,I,R = seir_discrete.sir_model(y, predicted_days, 
                            predicted_beta, gamma, 
                            stype, beta_t).T
        E = np.zeros((1,predicted_days.shape[0]))
    return S,E,I,R


def main_f(I_prediction_method, stochastic, count_stoch_line, 
           beta_prediction_method, start_day, seed_numbers):
    '''
    Основная функция
    
    Параметры:
    
    - I_prediction_method -- математическая модель для построения 
        траектории Infected
        ['seir', 'sir']
    - stochastic -- индикатор присутствия предсказанных стохастической 
        мат. моделью траекторий Infected 
    - count_stoch_line -- количество предсказанных стохастической 
        мат. моделью траекторий Infected 
    - beta_prediction_method -- метод предсказания значений Beta
        ['mean_const','expanding_mean','bi_exp_decay', 'polynom_of_day', 
        'polynom_of_day_prev_I','polynom_of_day_with_shift', 'polynom_SGDReg', 
        'polynom_SGDReg_with_rollingshift','polynom_SGDReg_with_future_training','lstm']
    - start_day -- день переключения на мат. модель
    - seed_numbers -- номера seed для экспериментов
    
    Выход:
        График для сидов.
    '''
    # устаноавление всегда постоянных значений параметров мат. модели
    sigma = 0.1
    gamma = 0.08
    
    fig, axes = plt.subplots(len(seed_numbers)//2+math.ceil(len(seed_numbers)%2), 
                             2, figsize=(15, 
                                         4*len(seed_numbers)//2+math.ceil(len(seed_numbers)%2)
                                        )
                            )
    axes = axes.flatten()
    
    # объявление папки с DataFrames of seeds, созданными регулярной сетью
    seed_dirs=f'{I_prediction_method}_30_seeds_v0/'
    
    # список RMSE Beta и I для каждого seed, чтобы изобразить boxplot
    all_rmse_I = []
    all_rmse_Beta = []
    peaks = []
    
    for idx, seed_number in enumerate(seed_numbers):
        
        # чтение DataFrame of seed: S,[E],I,R,Beta
        seed_df = pd.read_csv(seed_dirs + f'{I_prediction_method}_seed_{seed_number}.csv')
        seed_df = seed_df[pd.notna(seed_df['Beta'])]

        # выбор дней для предсказания
        predicted_days = np.arange(start_day, seed_df.shape[0])
        
        # предсказание значений Beta и подсчет времени этого процесса
        predicted_beta, execution_time, \
            predicted_I = predict_beta(I_prediction_method, seed_df, 
                                       beta_prediction_method, predicted_days, 
                                       stochastic, count_stoch_line, sigma, gamma)

        if beta_prediction_method != 'polynom_of_day_prev_I':
            # извлечение значений компартментов в день переключения на мат. модель
            y = seed_df.iloc[predicted_days[0]].drop('Beta')

            # предсказание траектория компартмента Infected
            _,_,predicted_I[0],_ = predict_I(I_prediction_method, y, 
                                    predicted_days, 
                                    predicted_beta, sigma, gamma, 'det')
            if stochastic:
                for i in range(count_stoch_line):
                    _,_,predicted_I[i+1],_ = predict_I(I_prediction_method, y, 
                                                predicted_days, 
                                                predicted_beta, sigma, gamma, 'stoch')
        

        # построение графика для seed_number
        ax = axes[idx]
        rmse_I, rmse_Beta, peak_h, pred_peak_h, \
            peak_t, pred_peak_t = plot_one(ax, predicted_days, seed_df, 
                                     predicted_I, predicted_beta, seed_number, 
                                     execution_time)        
    
        all_rmse_I.append(rmse_I)
        all_rmse_Beta.append(rmse_Beta)
        peaks.append([peak_h, pred_peak_h, peak_t, pred_peak_t])
    
    # добавление общего заголовка
    fig.suptitle(f'Switch {start_day} day, \n'+
                 f'I_prediction_method:{I_prediction_method}, \n'+
                 f'beta_prediction_method: {beta_prediction_method}' ,fontsize=15)
    plt.tight_layout()
    
    #сохранение графиков в pdf
    #plt.savefig(f'plots/{beta_prediction_method}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    #pd.DataFrame(peaks).to_csv(f'peak_metrics/{beta_prediction_method}_peaks.csv')
    
    plt.show() 
'''
    #построение графиков boxplot для RMSE Beta и I
    fig_box, ax1 = plt.subplots(figsize=(10, 6))
    bp1 = ax1.boxplot(all_rmse_I, positions=[1], widths=0.6, patch_artist=False)
    for median in bp1['medians']:
     median.set_color('blue')
    ax1.set_ylabel("Infected RMSE", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    bp2 = ax2.boxplot(all_rmse_Beta, positions=[2], widths=0.6,patch_artist=False)
    for median in bp2['medians']:
        median.set_color('green')
    ax2.set_ylabel("Beta RMSE", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax1.set_xlim(0.5, 2.5)
    ax1.set_xticks([1,2])
    ax1.set_xticklabels(["Infected RMSE", "Beta RMSE"])
    ax1.set_title("RMSE distributions across seeds")
    plt.tight_layout()
    plt.show() 
''';