#%%
import pandas as pd
import numpy as np
import datetime
from time import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#%%
# Data Loading
xente = pd.read_csv("../data/training.csv")
xente_test = pd.read_csv("../data/test.csv")

#%%
# Date transformation and date variable creation

xente.TransactionStartTime = pd.to_datetime(xente.TransactionStartTime)

xente['TransactionStartYear']= xente.TransactionStartTime.map(lambda date: date.year)
xente['TransactionStartMonth']= xente.TransactionStartTime.map(lambda date: date.month)
xente['TransactionStartDay'] = xente.TransactionStartTime.map(lambda date: date.day)
xente['TransactionStartHour'] = xente.TransactionStartTime.map(lambda date: date.hour)
xente['TransactionStartMinute'] = xente.TransactionStartTime.map(lambda date: date.minute)
xente['TransactionStartSecond'] = xente.TransactionStartTime.map(lambda date: date.second)

#%%
train = xente.drop(["FraudResult","TransactionStartTime","CurrencyCode","TransactionId"],axis=1)
y = xente.FraudResult

#%%
# SImple Random Forest Classifier
model_rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    max_depth=4)

#%%
nfolds = 5
#test_pred = np.zeros((test.shape[0], 2))
pred_oof = np.zeros((train.shape[0], 2))
scores, times = np.zeros(nfolds), np.zeros(nfolds)
i=0
kf= StratifiedKFold(nfolds, shuffle=True, random_state=129)
for (dev_index, val_index) in kf.split(train, y):
    print (f'\n**********  starting fold {i+1} *********')
    t1 = time()      
    xtr, xte = train[train.index.isin(dev_index)], train[train.index.isin(val_index)]
    ytr, yte = y[dev_index], y[val_index] 
    
    model_rf.fit(xtr, ytr)
    pred = model_rf.predict(xtr)
    test_pred_cv=model_rf.predict(xte)

    test_pred += test_pred_cv / nfolds
    pred_oof[val_index] = pred
    scores[i], times[i] = f1_score(yte, test_pred), round((time()-t1), 0)
    print(f'Fold {i+1} | score = {scores[i]} | time = {times[i]} s')
    i+=1

#%%
total_score = competition_scorer(y, pred_oof)
print(f'\nTotal score = {total_score} | mean = {round(np.mean(scores), 4)} | std = {round(np.std(scores), 4)}')


#%%
