import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sm
from sklearn import preprocessing
from scipy import stats

def normalisasiData(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset_minmax = min_max_scaler.fit_transform(dataset)

    return dataset_minmax

dataset = pd.read_csv('glcm_libari.csv', skiprows=[0], header=None)

class_data = dataset[8]
del dataset[8]

dataset = dataset.values
class_data = class_data.values

x_train =[]
x_test = []
y_train = []
y_test = []
train = []
test = []
dataset = stats.zscore(dataset)

skf = StratifiedKFold(n_splits=10)
clf = KNeighborsClassifier(n_neighbors= 6)
for train_index, test_index in skf.split(dataset, class_data):
    X_train, X_test = dataset[train_index], dataset[test_index]
    y_train, y_test = class_data[train_index], class_data[test_index]
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    model = clf.fit(X_train, y_train)
    model_result = model.predict(X_test)
    akurasi = (float(sm.accuracy_score(model_result, y_test)) * 100)
    presisi = (float(sm.precision_score(model_result, y_test)) * 100)
    print akurasi