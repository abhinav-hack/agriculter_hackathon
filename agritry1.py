# Using xgboost

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers
from xgboost import XGBClassifier

# loading data

df = pd.read_csv('/root/Documents/crop/train_yaOffsB.csv')
print(df.tail())
print(df.shape)

# filling na values

df.isnull().sum()
df.Number_Weeks_Used.fillna(0, inplace=True)


# poping out output
Crop_Damage = df.pop('Crop_Damage')
out_count = len(Crop_Damage.unique())

# cleaning unique id removing F
id = df['ID']
id = [hold[1:] for hold in id]
df.pop('ID')
df.insert(0, 'ID', id)

'''
#converting data to array
x_train = df.to_numpy()
y_train = Crop_Damage.to_numpy()

# Create a model

model = XGBClassifier(booster='gbtree', objective='multi:softmax',
    learning_rate=0.1, eval_metric="auc",
    max_depth=5, subsample=0.9, 
    colsample_bytree=0.8, num_class=out_count,
    n_estimators=200, gamma=0)

%time model.fit(x_train, y_train, verbose=True)

val = model.predict(x_train)

print(accuracy_score(y_train, val))
'''