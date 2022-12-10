# DATASET FROM: https://archive.ics.uci.edu/ml/datasets/Iris

# =========================== FOR IMPORTING THE DATASET ===========================

import numpy as np
import pandas as pd

from sklearn.preprocessing import Normalizer

# Added an ID column since CSV files have it while .data.text files do not
colnames=["SepalLengthCm", "SepalWidthCm","PetalLengthCm","PetalWidthCm", "Species","Id"]
dataset = pd.read_csv("iris.data", header = None, names= colnames )

dataset.groupby('Species').size()

# Nameing the features columns(X) from the corresponding plant name column(y)
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
# NOT Stratified
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 5)
# Stratified    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True,random_state =1, stratify=y)

    
#Normalizing the dataset
scaler= Normalizer().fit(X_train) # the scaler is fitted to the training set
normalized_X_train= scaler.transform(X_train) # the scaler is applied to the training set
normalized_X_test= scaler.transform(X_test) # the scaler is applied to the test set
print('x train before Normalization')
print(X_train[0:5])
print('\nx train after Normalization')
print(normalized_X_train[0:5])

# =========================== FOR STARTING THE ALGORITHM ===========================

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Instantiate learning model
# Using the Nearest-Neighbor
my_classifier = KNeighborsClassifier(n_neighbors=3)
# Using the Euclidean Distance Method
# my_classifier = KNeighborsClassifier(n_neighbors=3,metric='euclidean')

# Fitting the model
# For NON-Normalized Values
# my_classifier.fit(X_train, y_train)
# For Normalized Values
my_classifier.fit(normalized_X_train, y_train)

# Predicting the Test set results
# For NON-Normalized Values         
# my_y_pred = my_classifier.predict(X_test)
# For Normalized Values
my_y_pred = my_classifier.predict(normalized_X_test)

# =========================== FOR RESULTS ===========================

mse = mean_squared_error(y_test, my_y_pred)*100             
print('Accuracy of our model is equal ' + str(round(mse, 2)) + ' %.')

accuracy = accuracy_score(y_test, my_y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')