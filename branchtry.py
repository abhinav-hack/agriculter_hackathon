# trying without id column

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold

# loading data

df = pd.read_csv('/root/Documents/crop/train_yaOffsB.csv')
print(df.tail())
print(df.shape)

#filling na values    #### no need of it xgboost will handle it
#df.isnull().sum()
#df.Number_Weeks_Used.fillna(0, inplace=True)




Crop_Damage = df.pop('Crop_Damage')
out_count = len(Crop_Damage.unique())

# cleaning unique id removing F
id = df['ID']
extra  = [hold[1:] for hold in id]
df.pop('ID')
df.insert(0, 'ID', extra)

print(df.shape)

#converting data to array
x_train = df.to_numpy()
y_train = Crop_Damage.to_numpy()

x_validate, x_train =  x_train[:17771], x_train[17771:]
y_validate, y_train =  y_train[:17771], y_train[17771:]


# Create a model  // trying different model


#
#
model = XGBClassifier(booster='gbtree', objective='multi:softmax',
    learning_rate=0.9, eval_metric="auc",
    max_depth=6, subsample=0.6, colsample_bylevel=0.6,
    colsample_bytree=0.7, num_class=out_count,
    n_jobs=6,
    max_delta_step = 3, min_child_weight=0.01,
    n_estimators=300, gamma=1, alpha= 4.8)

model.fit(x_train, y_train, verbose= True)

print('train acc')
pred = model.predict(x_train)
print(accuracy_score(y_train, pred))

print('validation acc')
valpred = model.predict(x_validate)
print(accuracy_score(y_validate, valpred))
# trying without id column

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

# loading data

df = pd.read_csv('/root/Documents/crop/train_yaOffsB.csv')
print(df.tail())
print(df.shape)

#filling na values    #### no need of it xgboost will handle it
#df.isnull().sum()
#df.Number_Weeks_Used.fillna(0, inplace=True)

# poping out output and index

Crop_Damage = df.pop('Crop_Damage')
out_count = len(Crop_Damage.unique())

id = df.pop('ID')
print(df.shape)

#converting data to array
x_train = df.to_numpy()
y_train = Crop_Damage.to_numpy()


x_train, x_validate = x_train[:71085], x_train[71085:]
y_train, y_validate = y_train[:71085], y_train[71085:] 

# Create a model

model = XGBClassifier(booster='gbtree', objective='multi:softmax',
    learning_rate=0.5, eval_metric="auc",
    max_depth=14, subsample=0.7, colsample_bylevel=0.6,
    colsample_bytree=0.6, num_class=out_count,
    max_delta_step = 1,
    n_estimators=300, gamma=3, alpha= 3)

model.fit(x_train, y_train, verbose= True)

print('train acc')
pred = model.predict(x_train)
print(accuracy_score(y_train, pred))

print('validation acc')
valpred = model.predict(x_validate)
print(accuracy_score(y_validate, valpred))
