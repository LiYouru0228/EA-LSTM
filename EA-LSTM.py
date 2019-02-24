# coding: utf-8

import random
from itertools import combinations
from __future__ import division 
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)
from pandas import read_csv
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error,mean_absolute_error
import scipy as sp
import numpy as np

# Initialize random population
def get_first_pop(pop_size, param_num, encode_length):
    pop = []
    for i in range(pop_size):
        individual = []
        for i in range(param_num * encode_length):
            temp = random.randint(0,9)
            if temp > 4:
                individual.append(1)
            else:
                individual.append(0)
        pop.append(individual)
    return pop

# Decoding parameter
def encode(individual_code):
    temp = [str(each) for each in individual_code] 
    # normalization
    result = round(int(''.join(temp),2)/63,2)
    # exception handling
    if result == 0.0:
        result += 0.01
    return result

# Decoding parameter combination
def get_param(individual_code,param_num, encode_length):
    param = []
    for i in range(param_num):
        param.append(encode(individual_code[i*encode_length:(i+1)*encode_length]))
    return param

# Decoding population
def pop2param(pop):
    param = []
    for each in pop:
        param.append(get_param(each,18,6))
    return param

# Initialization parameter population
def get_first_param(pop_size, param_num, encode_length):
    first_pop = get_first_pop(pop_size, param_num, encode_length)
    param = []
    for each in first_pop:
        param.append(get_param(each,param_num,encode_length))
    result = []
    result.append(first_pop)
    result.append(param)
    return result

# Get random index
def get_cross_seg_id(param_num):
    index = []
    while(1):
        for i in range(param_num):
            temp_r = random.randint(0,9)
            if temp_r > 4:
                index.append(i)
        if len(index) > 0:
            return index
        else:
            continue

# Crossover
def cross_over(individual_1,individual_2): 
    index = get_cross_seg_id(108)
    c_index = []
    for i in range(108):
        if i not in index:
            c_index.append(i)
    
    new = []
    for i in range(108):
        new.append(0)
    
    for each in index:
        new[each] = individual_1[each]
    for each in c_index:
        new[each] = individual_2[each]
    
    if random.randint(0,9) > 7:
        new = variation(new)
    
    return new

# variation 
def variation(individual_code):
    index = get_cross_seg_id(random.randint(1,108))
    for each in index:
        if individual_code[each] == 0: 
            individual_code[each] = 1
        else:
            individual_code[each] = 0
    return individual_code

# Population grouping
def pop2group(pop,group_num):
    group_index = random.sample(range(0,group_num*6),group_num*6)
    group_pop = []
    for i in range(group_num):
        temp_index = group_index[i*group_num:(i+1)*group_num]
        temp_pop = []
        for each in temp_index:
            temp_pop.append(pop[each])
        group_pop.append(temp_pop)
    return group_pop

# Conversion of parameter sequences to key values
def c_pop2str(c_pop):
    temp = [str(each) for each in c_pop] 
    key = ''.join(temp)
    return key

# select
def select(pop,n_selected,step):
    group_pop = pop2group(pop,n_selected)
    fitness_selected = []
    pop_selected = []
    c1 = 0
    for each_group in group_pop:
        fit_temp = []
        c1 += 1
        c2 = 0
        for each in each_group:
            c2 += 1 
            print '----------------------------------------------------------'
            print 'Step:',step,'-',c1,'-',c2
            c_pop = c_pop2str(each)
            if c_pop2rmse.has_key(c_pop):
                fit_temp.append(c_pop2rmse[c_pop])
                print 'Get RMSE from Hash...'
            else:
                rmse = get_rmse(get_param(each,18,6),reframed)
                fit_temp.append(rmse)
                c_pop2rmse[c_pop] = rmse
        pop_selected.append(each_group[fit_temp.index(min(fit_temp))])
        fitness_selected.append(min(fit_temp))
            
    selected = []
    selected.append(pop_selected)
    selected.append(fitness_selected)
    return selected

# Population reconstruction
def pop_reconstruct(pop_selected,target_num):
    new_pop = []
    new_pop.extend(pop_selected)
    
    temp_pop_map = {}
    for i in range(len(new_pop)):
        temp_pop_map[c_pop2str(new_pop[i])] = i
        
    index = [c for c in  combinations(range(len(pop_selected)), 2)]
    while(len(new_pop) < target_num):
        for each in index:
            new = cross_over(pop_selected[each[0]],pop_selected[each[1]])
            if temp_pop_map.has_key(c_pop2str(new)) == False:
                new_pop.append(new)
                temp_pop_map[c_pop2str(new)] = len(new_pop)
            if len(new_pop) == target_num:
                return new_pop
    return  new_pop   

# preprocessing
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    #(t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    
    #(t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t + %d)' % (j + 1, i)) for j in range(n_vars)]
    
    # concat
    agg = concat(cols, axis = 1)
    agg.columns = names
    
    # dropnan
    if dropnan:
        agg.dropna(inplace = True)
    return agg

# produce attention weights 
def get_attenton_rate_df(param,df,n_feature,time_step):
    attenton_rate = param
    print 'attenton_rate:',attenton_rate
    new_reframed = df
    index = []
    for i in range(time_step):
        for j in range(n_feature):
            index.append('var'+str(j+1)+'(t-'+str(time_step-i)+')')

    for i in range(n_feature):
        if i == 0:
            for j in range(time_step):
                new_reframed[index[n_feature*i + j]] *= attenton_rate[i]
    return new_reframed

# Get the training error of the attention weight on the validation set
def get_rmse(param,reframed):
    
    # Training set and validation set data
    values = get_attenton_rate_df(param,reframed,8,18).values
    n_train_hours = int(35040*0.8)
    train = values[:n_train_hours, :]
    valid = values[n_train_hours:35040, :]
    
    # specify the number of lag hours
    n_hours = 18
    n_features = 8

    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    valid_X, valid_y = valid[:, :n_obs], valid[:, -n_features]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    valid_X = valid_X.reshape((valid_X.shape[0], n_hours, n_features))
    
    # design model
    model = Sequential()
    model.add(LSTM(128, input_shape = (train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(LSTM(128,return_sequences = False))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'adam')

    # training
    history = model.fit(train_X, train_y, epochs = 100, batch_size = 256, validation_data = (valid_X, valid_y), verbose = 1, shuffle = True)

    # make a prediction
    yhat = model.predict(valid_X)
    valid_X = valid_X.reshape((valid_X.shape[0], 18*8))

    # inverse transform of prediction
    inv_yhat = concatenate((yhat, valid_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # inverse transform of real value
    valid_y = valid_y.reshape((len(valid_y), 1))
    inv_y = concatenate((valid_y, valid_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # metric
    rmse = sp.sqrt(mean_squared_error(inv_y, inv_yhat))
    print 'The RMSE:',rmse
    mae = mean_absolute_error(inv_y, inv_yhat)
    print 'The MAE:',mae
    
    return rmse,mae

def search_best_attention_rate(max_step):
    step = 0
    while (step != max_step):
        print 'Epoch:',step 
        
        # initialize  
        if step == 0:
            value = get_first_param(36,18,6)
            pop = value[0]
        
        # select
        selected = select(pop,6,step)
        
        # reconstruct
        pop = pop_reconstruct(selected[0],36)
        
        # deeper search
        step += 1
        
# Loading preprocessed data
base_path = '.../dataset/BeijingPM2.5/'
dataset = read_csv(base_path + 'pollution.csv', header=0, index_col=0)

# load values
values = dataset.values

# Categorical variable processing
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
dataset['wnd_dir'] = values[:,4] 
dataset.to_csv(base_path + 'beijing_pm2.5.csv')

# Unified data format as float32
values = values.astype('float32')

# Normalized
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(values)

# Sliding window
reframed = series_to_supervised(scaled, 18, 1)
index_col = 18 * 8 

# Remove useless columns
reframed.drop(reframed.columns[[index_col+1,index_col+2,index_col+3,index_col+4,index_col+5,index_col+6,index_col+7]], axis = 1, inplace = True)

# save results
c_pop2rmse = {}
step_param = []
step_fitness = []

# let's begin
search_best_attention_rate(20)

