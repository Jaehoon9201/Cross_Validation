# Load libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from keras.datasets import mnist
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
import keras as K
import keras
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import glob
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

X= np.array([[1, 0, 0.8, 1] ,  [0.8, 0, 1, 1]  , [1, 0, 1, 0.8  ], [1, 0, 0.8, 0.8] , [0.8, 0, 0.8, 0.8],
            [1.1, 0, 1.3, 1],  [1.4, 0, 1, 1]  , [1, 0, 1.3, 1]  , [1, 0, 1.2, 1]   , [1.1, 0, 1.1, 1],
            [0, 0, 1, 1]    ,  [0, 0, 1, 1]    , [0, 0, 1, 1]    , [0, 0, 1, 1]     , [0, 0, 1, 1],
            [0, 0, 1.3, 1]  , [0, 0, 1, 1]     , [0, 0, 1.3, 1]  , [0, 0, 1.2, 1]   , [0, 0, 1.1, 1],
            [0.7, 0, 1, 1]  , [0.7, 0, 0.7, 1] , [0.5, 0, 1, 1]  , [0.5, 0, 0.5, 1] , [0.6, 0, 1, 0.5],
            [0, 0, 1.6, 1]  , [0, 0, 1, 1.6]   , [0, 0, 1.3, 1.6], [0, 0, 1.2, 1.6] , [0, 0, 1.1, 1.6],
            [1.7, 0, 1, 1]  , [1.7, 0, 0.7, 1] , [1.5, 0, 1, 1]  , [1.5, 0, 0.5, 1] , [1.6, 0, 1, 0.5],
            [0, 0, 1.6, 1.2], [0, 0, 1, 1.8]   , [0, 0, 1.3, 1.8], [0, 0, 1.2, 1.8] , [0, 0, 1.4, 1.6]])
y= [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
    [1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
    [0],[0],[0],[0],[0],[1],[1],[1],[1],[1],
    [0],[0],[0],[0],[0],[1],[1],[1],[1],[1]]
# ■■■■■■■■■■■■■■■■■■
# ■■■■■■   NN   ■■■■■■■
# ■■■■■■■■■■■■■■■■■■
y_cat = to_categorical(y)

dense1 = 6
dense2 = 6
train_epoch = 60
batch_size = 4  # 340
classes = 2
learn_rate = 1e-4
original_dim =4

def create_model():
    model = Sequential()
    model.add(Dense(dense1, input_dim = original_dim, activation = 'relu'))
    model.add(Dense(dense2, activation='relu'))
    model.add(Dense(classes,input_dim=original_dim, activation='softmax'))
    #model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn = create_model, epochs = 120, batch_size = None, verbose = 0)
Kfold = KFold(n_splits = 2, shuffle = True, random_state = seed)
results = cross_val_score(model, X, y_cat, cv = Kfold)
print(results)