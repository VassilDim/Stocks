## Import libraries:
# Stats:
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

# System:
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings

# plotting
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

## FUNCTIONS:

def sarima_forecast (history, config):
    order, sorder, trend = config
    # Define model
    model = SARIMAX(history,
                    order=order,
                    seasonal_order=sorder,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                    )
    # Fit model
    print('Fitting model...')
    model_fit = model.fit(disp=False, maxiter=5000)
    # Make a 1-step forecast
    print('Making prediction...')
    yhat = model_fit.predict(len(history), len(history)+1)
    print('Prediction:', yhat[0])
    return yhat[0]

def train_test_split (data, n_test):
    print('train_test_split...')
    return data[:-n_test], data[-n_test:]

def measure_rmse (actual, predicted):
    print('Calculating error...')
    return sqrt(mean_squared_error(actual, predicted))

def walk_forward_validation (data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split (data, n_test)
    # seed history with training data
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range (len(test)):
        print(f'Making predictions number {i}')
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for next iteration
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error

# score a model, return None if failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show warning and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggest unstable config
        try:
            # never show warning when grid searching
            result = walk_forward_validation(data, n_test, cfg)
        except Exception as e:
            print (f'An error occured: {e}',)
            result = None

            #with catch_warning():
            #    filterwarnings('ignore')
            #    result = walk_forward_validation(data, n_test, cfg)
        #except:
        #    error = None
    # check for results
        print('\n------------', (key, result), '\n------------\n\n')
    #if result is not None:
    #    print((key, result))
    return (key, result)

def grid_search(data, cfg_list, n_test):
    scores = []
    for cfg in cfg_list:
        print(cfg)
        score = score_model(data, n_test, cfg)
        scores.append(score)
    scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    #scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# Create SARIMA configurations to be tested
def sarima_configs(seasonal=[5, 20]): # seasonality of week and year
    models = list()
    # define config lists
    p_params = [1, 2] # 1 discovered from PACPlot
    d_params = [0] # 0 from using days
    q_params = [0, 1, 2] # Moving average - nothing or last 2 months
    t_params = ['ct','c', 'n'] # The trend is non of these but try
    P_params = [0, 1, 2, 4] # Seasonal autoregressive order (weekly)
    D_params = [0] # Seasonal differencing
    Q_params = [0, 1] # Seasonal adjustment
    m_params = seasonal # Seasonality
    # Create config instances:
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg=[(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models


## Load Data
df = pd.read_pickle('df_clean_2020.pkl')
data = list(df['close'])
# data split
n_test = 10
# Generating hyperparameter combinations
cfg_list = sarima_configs()
print('Total number of configurations to test', len(cfg_list),\
'\n------------\n')
# grid search
scores = grid_search(data, cfg_list, n_test)
# List top 3 configurations:
for cfg, error in scores[:3]:
    print (cfg, error)

'''
[(1, 0, 1), (1, 0, 1, 5), 'ct'] 0.046184592185630854
[(1, 0, 1), (2, 0, 1, 5), 'ct'] 0.04874532046906347
[(2, 0, 2), (1, 0, 1, 5), 'ct'] 0.05035231078342537
'''
