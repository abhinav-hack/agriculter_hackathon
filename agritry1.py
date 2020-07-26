# Using xgboost train

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

# filling na values      ## no need of it xgboost will handle it

#df.isnull().sum()
#df.Number_Weeks_Used.fillna(0, inplace=True)


#poping out output 
Crop_Damage = df.pop('Crop_Damage')
out_count = len(Crop_Damage.unique())

# cleaning unique id removing F
id = df['ID']
id = [hold[1:] for hold in id]
df.pop('ID')
df.insert(0, 'ID', id)


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


model.fit(x_train, y_train, verbose=True)
#print('preparing for kfold')

#k_fold = KFold(len(x_validate), n_splits=3, shuffle=True, random_state=0)
#print(np.mean(cross_val_score(model, x_validate, y_validate, cv=k_fold, n_jobs=1))) 

print('train acc')
pred = model.predict(x_train)
print(accuracy_score(y_train, pred))

print('validation acc')
valpred = model.predict(x_validate)
print(accuracy_score(y_validate, valpred))

