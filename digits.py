import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, ParameterGrid
from sklearn.utils import shuffle
import math
import time
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RANDOM_STATE = 1


def getXy(data):
    y = data.loc[:, 'label']
    col_names = list(data.columns)
    col_names.remove('label')
    X = data.loc[: , col_names]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)    
    return X, y

def extract_data(data, n):
    data = shuffle(data, random_state = RANDOM_STATE)
    train, test = train_test_split(data, test_size = 0.2, random_state = RANDOM_STATE)
    train = train.sample(n, random_state=RANDOM_STATE)
    X_train, y_train = getXy(train)
    X_test, y_test = getXy(test)
    
    return [{'train': X_train, 'test': X_test}, {'train': y_train, 'test': y_test}]

def trainer_by_time(X, y, time_limits, nn):
    running_time = 0
    evaluation_list = []
    i = 0
    num_itr = 0
    while True:
        start_time=time.process_time()
        nn.fit(X['train'], y['train'])
        end_time=time.process_time()
        num_itr += 1
        running_time += end_time - start_time
        if running_time > time_limits[i]:
            evaluation = evaluate_model(X, y, nn)
            evaluation['running time'] = running_time
            evaluation['num_itr'] = num_itr
            evaluation_list.append(evaluation)
            i += 1
        if i == len(time_limits):
            return evaluation_list

def evaluate_model(X, y, nn):
    result = {}
    for part in ('train', 'test'):
        y_pred = nn.predict(X[part])
        result[part + '_accuracy'] = accuracy_score(y[part], y_pred)
        y_pred_prob = nn.predict_proba(X[part])
        result[part + '_log_loss'] = log_loss(y[part], y_pred_prob)
    
    #cv_accuracy = cross_val_score(nn, X, y, scoring='accuracy', cv=3).mean()
    #cv_logloss = cross_val_score(nn, X, y, scoring='log_loss', cv=3).mean()    
    #return {'train': {'accuracy': training_accuracy, 'log_loss': training_logloss},
            #'test': {'accuracy': cv_accuracy, 'log_loss': cv_logloss}}
    return result


#def main1():
    #for n in [200]:
        #sample = data.sample(n, random_state=1)
        #X, y = extract_data(sample)
        #activation = 'tanh'
        ##for activation in ['relu','logistic', 'tanh']:
        #nn = MLPClassifier(activation=activation, tol=float(-'inf'), warm_start = True, max_iter=1)
         ##print( activation, 'n=', n, modeling(sample,nn), file=result_file)
        #time_limits = [30]
        #evaluation_list = None
        
        #for item in ['train', 'test']:
            #record = {}
            #record['n'] = n
            #record['activation'] = activation
            #record['train/test'] = item
            #for metric in metrics:
                
                    #record[metric] = evaluate_model(sample, nn)[item][metric]  
            #records.append(record)    
            
def main():
    np.random.seed(RANDOM_STATE)
    pd.set_option('display.width', 0)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    data = pd.read_csv('data/train.csv')
    
    #test_data = pd.read_csv('data/test.csv')
    
    records = []
    
    #n = 42000*0.8
    n = 10000
    X, y = extract_data(data, n)
    activation = 'tanh'
    param_dict = {'batch_size': [100, 200], 'momentum': [0.9, 0.99 ], 'learning_rate_init':[0.001, 0.01, 0.1]}
    
    for param in ParameterGrid(param_dict):       
        nn = MLPClassifier(algorithm='sgd', 
                           tol=float('-inf'),
                           warm_start = True,
                           max_iter=1, 
                           hidden_layer_sizes = [200],
                           random_state=RANDOM_STATE)
        #nn_params = {'algorithm': 'sgd', 'tol': float
        nn_params = nn.get_params()
        nn_params.update(param)
        nn.set_params(**nn_params)
        #nn = MLPClassifier(**nn_params)
        time_limits = list(range(60, 600, 60))
        evaluation_list = trainer_by_time(X, y, time_limits, nn)
        for i in range(len(evaluation_list)):
            evaluation = evaluation_list[i]
            record = {}
            record['n'] = n
            record['time limit'] = time_limits[i]
            record.update(evaluation)  
            record.update(param)
            records.append(record)
        
        
    df = pd.DataFrame(records)
    cols = list(df.columns)
    keys = evaluation_list[0].keys()
    cols = [item for item in cols if item not in keys]
    cols += keys
    df = df.reindex(columns=cols)
    result_file = open('result.txt', 'a')
    print(df)
    print(df,file=result_file)
    
    
main()