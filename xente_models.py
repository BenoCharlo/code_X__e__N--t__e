#%%
import warnings; warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import datetime
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette='pastel')
sns.set(font_scale=2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

from utils_funcs import MultiColumnLabelEncoder, date_transformer, train_test_together

#%%
# Data Loading
xente = pd.read_csv("../data/training.csv")
xente_test = pd.read_csv("../data/test.csv")

#%%
# Date transformation and date variable creation
date_variable="TransactionStartTime"

xente=date_transformer(xente, date_variable)
xente_test=date_transformer(xente_test, date_variable)

#%%
train = xente.drop([
    "FraudResult","TransactionStartTime","CurrencyCode",
    "TransactionId", "CountryCode"],axis=1)
y = xente.FraudResult

test = xente_test.drop([
    "TransactionStartTime","CurrencyCode",
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
# Value_Amount diff

xente_all['Value_Not_Amount'] = (xente_all.Amount == xente_all.Value)*1

xente_all['Value_Amount_Diff'] =  np.abs(xente_all.Amount) - np.abs(xente_all.Value)

#%%
# Re-separate the data

train=xente_all.loc[xente_all.is_train==1]
train.drop("is_train",axis=1,inplace=True)

test=xente_all.loc[xente_all.is_train==0]
test.drop("is_train",axis=1,inplace=True)

#%%
#plt.hist(xente.loc[xente.FraudResult == 1, "Amount"], bins='auto', alpha=0.7, label='1', color='b')
plt.boxplot(xente.loc[xente.FraudResult == 1, "Amount"]) #bins=20, alpha=0.7, label='0' , color='r'
plt.title("Histogram of Amount")
plt.legend(loc='upper right')
plt.show()

#%%
# Simple Random Forest Classifier
model_rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    max_depth=6)

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
    max_depth=6)

model_rf.fit(train,y)

#%%
features = list(train.columns)
importances = model_rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

#%%
model_rf = RandomForestClassifier(
    n_estimators=250,
    n_jobs=-1,
    max_depth=6)

model_rf.fit(train,y)

test_pred = model_rf.predict(test)

#%%
submission=pd.DataFrame({
    "TransactionId":xente_test.TransactionId.tolist(),
    "FraudResult":list(test_pred)})

submission.to_csv("../submissions/RF_3.csv",index=False)

#%%
# Simple Linear Models

model_lr = LogisticRegression(random_state=0, class_weight="balanced")

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

    model_lr.fit(xtr, ytr)
    pred = model_lr.predict(xtr)
    test_pred = model_lr.predict(xte)
    pred_oof[train_index] = np.reshape(pred,(len(train_index),1))
    scores[i], times[i] = f1_score(yte, test_pred), round((time()-t1), 0)
    print(f'Fold {i+1} | score = {scores[i]} | time = {times[i]} s')
    i+=1

#%%
model_lr = LogisticRegression(random_state=0).fit(train,y)

test_pred = model_lr.predict(test)

#%%
submission=pd.DataFrame({
    "TransactionId":xente_test.TransactionId.tolist(),
    "FraudResult":list(test_pred)})

submission.to_csv("../submissions/LR_1.csv",index=False)


#%%
