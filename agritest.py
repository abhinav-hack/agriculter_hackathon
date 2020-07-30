# Using xgboost test

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from agritry1 import model


# loading data

df = pd.read_csv('/root/Documents/crop/test_pFkWwen.csv')
print(df.tail())
print(df.shape)

# filling na values     ## no need of it xgboost will handle it

#df.isnull().sum()
#df.Number_Weeks_Used.fillna(0, inplace=True)


# poping out output
#Crop_Damage = df.pop('Crop_Damage')
#out_count = len(Crop_Damage.unique())

# cleaning unique id removing F
id = df['ID']
extra  = [hold[1:] for hold in id]
df.pop('ID')
df.insert(0, 'ID', extra)


#converting data to array
x_test = df.to_numpy()
#y_train = Crop_Damage.to_numpy()

# Create a model

# Here is the model that imported from train agritry module

#model = XGBClassifier(booster='gbtree', objective='multi:softmax',
#    learning_rate=0.9, eval_metric="auc",
#    max_depth=9, subsample=0.8, colsample_bylevel=0.6,
#    colsample_bytree=0.7, num_class=out_count,
#     n_jobs=6,
#    max_delta_step = 1, min_child_weight=0.1,
#    n_estimators=300, gamma=0.2, alpha= 0.2)

#model.fit(x_train, y_train, verbose=True)

val = model.predict(x_test)

#print(accuracy_score(y_train, val))

output = pd.DataFrame()
output['ID'] = id
output['Crop_Damage'] = val
print(output)

output.to_csv('/root/Documents/crop/output.csv', index= False)


