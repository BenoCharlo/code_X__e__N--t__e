import pandas as pd
from sklearn import preprocessing


def date_transformer(data, date_variable):

    """ Takes in dataframe containing a date variable and
    creates new time variables like year, month, day, etc."""

    data[date_variable] = pd.to_datetime(data[date_variable])

    data['TransactionStartYear']= data[date_variable].map(lambda date: date.year)
    data['TransactionStartMonth']= data[date_variable].map(lambda date: date.month)
    data['TransactionStartDay'] = data[date_variable].map(lambda date: date.day)
    data['TransactionStartHour'] = data[date_variable].map(lambda date: date.hour)
    data['TransactionStartMinute'] = data[date_variable].map(lambda date: date.minute)
    data['TransactionStartSecond'] = data[date_variable].map(lambda date: date.second)

    return data

def train_test_together(train, test):
    """" Creates a new variable name 'is_train' in train and test, then 
    concatenate the two dataframes"""

    train["is_train"] = [1]*train.shape[0]
    test["is_train"] = [0]*test.shape[0]

    return pd.concat([train, test])



class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
