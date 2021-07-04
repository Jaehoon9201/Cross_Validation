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


# ■■■■■■■■■■■■■■■■■■
# ■■■■■■   NN   ■■■■■■■
# ■■■■■■■■■■■■■■■■■■

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

y_cat = to_categorical(y)

dense1 = 6
dense2 = 6
batch_size = 4  # 340
classes = 2
learn_rate = 1e-4
original_dim =4

def get_model():
    model = Sequential()
    model.add(Dense(dense1, input_dim = original_dim, activation = 'relu'))
    model.add(Dense(dense2, activation='relu'))
    model.add(Dense(classes,input_dim=original_dim, activation='softmax'))
    #model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

# Create the k-folds cross-validator
num_folds = 2
kfold = KFold(n_splits=num_folds, shuffle=True)

num_epochs = 120
train_scores = []
test_scores = []
fold_index = 1

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Train model for each fold
for train, test in kfold.split(X, y_cat):
    # Create model
    model = get_model()

    # Fit the model
    history = model.fit(
        X[train],
        y_cat[train],
        validation_data=(X[test], y_cat[test]),
        epochs=num_epochs,
        batch_size=4,
        verbose=0
    )

    # Preserve the history and print out some basic stats
    train_scores.append(history.history['accuracy'])
    test_scores.append(history.history['val_accuracy'])
    print("Fold %d:" % fold_index)
    print("Training accuracy: %.2f%%" % (history.history['accuracy'][-1] * 100))
    print("Testing accuracy: %.2f%%" % (history.history['val_accuracy'][-1] * 100))

    fold_index += 1


# Set up the plot
plt.figure()
plt.title('Cross-Validation Performance')
plt.ylim(0.2, 1.01)
plt.xlim(0, num_epochs - 1)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.grid()

# Calculate mean and distribution of training history
train_scores_mean = np.mean(train_scores, axis=0)
train_scores_std = np.std(train_scores, axis=0)
test_scores_mean = np.mean(test_scores, axis=0)
test_scores_std = np.std(test_scores, axis=0)

# Plot the average scores
plt.plot(
    train_scores_mean,
    '-',
    color="b",
    label="Training score"
)
plt.plot(
    test_scores_mean,
    '-',
    color="g",
    label="Cross-validation score"
)

# Plot a shaded area to represent the score distribution
epochs = list(range(num_epochs))
plt.fill_between(
    epochs,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="b"
)
plt.fill_between(
    epochs,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g"
)

plt.legend(loc="lower right")
# plt.show()
plt.show()