import cv2
import os
import glob
import pandas as pd
from os import listdir
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as sm
from statistics import mean
import numpy as np

def feature(img):
    xs=[]
    glcm = greycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'contrast')[0,0])
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    xs.append(greycoprops(glcm, 'homogeneity')[0, 0])
    xs.append(greycoprops(glcm, 'ASM')[0, 0])
    xs.append(greycoprops(glcm, 'energy')[0, 0])
    xs.append(greycoprops(glcm, 'correlation')[0, 0])
    return xs

def split_into_rgb_channels(image):
  red = image[:,:,2]
  green = image[:,:,1]
  blue = image[:,:,0]
  return red, green, blue
    
def features_extraction(datas):
    features = []
    for i in range(len(datas)):
        data = datas[i]
        features.append(feature(data))
    return features

def normalization(features):
    min_max = preprocessing.MinMaxScaler()
    x_scaled = min_max.fit_transform(features)
    dataset = pd.DataFrame(x_scaled)
    dataset = dataset.values
    return dataset

def KNN_model(dataset, class_data):
    X_train =[]
    X_test = []
    y_train = []
    y_test = []
    list_akurasi = []
    
    skf = StratifiedKFold(n_splits=3)
    clf = KNeighborsClassifier(n_neighbors= 3, weights='distance')
    for train_index, test_index in skf.split(dataset, class_data):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = class_data[train_index], class_data[test_index]
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        model = clf.fit(X_train, y_train)
        model_result = model.predict(X_test)
        akurasi = (float(sm.accuracy_score(model_result, y_test)) * 100)
        list_akurasi.append(akurasi)
        
    print mean(list_akurasi)
    
img_dir = "Dataset Wajah"

data_path = os.path.join(img_dir,'*.jpg')
files = glob.glob(data_path)
datas = []
class_data = []
redS = []
blueS = []
greenS = []
for f1 in files:
    img = cv2.imread(f1)
    red, green, blue = split_into_rgb_channels(img)
    redS.append(red)
    greenS.append(green)
    blueS.append(blue)
    head, tail = os.path.split(f1)
    name = tail.split('.')
    class_data.append(name[0])
    
redFeatures = features_extraction(redS)
greenFeatures = features_extraction(greenS)
blueFeatures = features_extraction(blueS)

datas = np.append(redFeatures, greenFeatures, axis=1)
datas = np.append(datas, blueFeatures, axis=1)

dataset = normalization(datas)

class_data = np.array(class_data)

KNN_model(dataset, class_data)


