#%%
import pandas as pd
import numpy as np
import datetime
from time import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils_funcs import MultiColumnLabelEncoder, date_transformer, train_test_together

#%%
# Data Loading
xente = pd.read_csv("../data/training.csv")
xente_test = pd.read_csv("../data/test.csv")

#%%
# Date transformation and date variable creation
date_variable="TransactionStartTime"

xente=date_transformer(xente, date_variable)
xente_test=date_transformer(xente, date_variable)

#%%
train = xente.drop([
    "FraudResult","TransactionStartTime","CurrencyCode",
    "TransactionId", "CountryCode"],axis=1)
y = xente.FraudResult

test = xente_test.drop([
    "FraudResult","TransactionStartTime","CurrencyCode",
    "TransactionId", "CountryCode"],axis=1)
#%%
# Data combining
xente_all = train_test_together(train, test)

#%%
# Label Encoding

multi_le = MultiColumnLabelEncoder(columns=[
    "BatchId","AccountId", "SubscriptionId","CustomerId", 
    "ProviderId", "ProductId", "ProductCategory",
    "ChannelId", "PricingStrategy"])

xente_all = multi_le.fit_transform(xente_all)

#%%
# Re-separate the data

train=xente_all.loc[xente_all.is_train==1]
train.drop("is_train",axis=1,inplace=True)

test=xente_all.loc[xente_all.is_train==0]
test.drop("is_train",axis=1,inplace=True)

#%%
# SImple Random Forest Classifier
model_rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    max_depth=4)

#%%
nfolds = 5
pred_oof = np.zeros((train.shape[0],1))
scores, times = np.zeros(nfolds), np.zeros(nfolds)

i=0
kf = StratifiedKFold(nfolds, shuffle=True, random_state=129)
for (train_index, valid_index) in kf.split(train, y):
    print(f'======= starting fold {i+1} ========')
    t1=time()
    xtr, xte = train[train.index.isin(train_index)], train[train.index.isin(valid_index)]
    ytr, yte = y[train_index], y[valid_index]

    model_rf.fit(xtr, ytr)
    pred = model_rf.predict(xtr)
    test_pred = model_rf.predict(xte)
    pred_oof[train_index] = np.reshape(pred,(len(train_index),1))
    scores[i], times[i] = f1_score(yte, test_pred), round((time()-t1), 0)
    print(f'Fold {i+1} | score = {scores[i]} | time = {times[i]} s')
    i+=1

#%%
total_score = f1_score(y, pred_oof)
print(f'\nTotal score = {total_score} | mean = {round(np.mean(scores), 4)} | std = {round(np.std(scores), 4)}')

#%%
model_rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    max_depth=4)

model_rf.fit(train,y)

test_pred = model_rf.predict(test)

#%%
submission=pd.DataFrame({
    "TransactionId":xente_test.TransactionId.tolist(),
    "FraudResult":list(test_pred)})

submission.to_csv("../submissions/RF_1.csv",index=False)

#%%
