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

# Create a model

model = XGBClassifier(booster='gbtree')
