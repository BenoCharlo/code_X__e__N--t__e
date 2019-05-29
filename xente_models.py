#%%
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#%%
# Data Loading
xente = pd.read_csv("data/training.csv")

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
