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
    
    
img_dir = "F:\Semester 7\Visikom\Classification 1\Dataset Wajah"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
datas = []
class_data = []
features = []
for f1 in files:
    img = cv2.imread(f1)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    datas.append(gray_image)
    head, tail = os.path.split(f1)
    name = tail.split('.')
    class_data.append(name[0])
    
for i in range(len(datas)):
    data = datas[i]
    features.append(feature(data))
    
#X_train, X_test, y_train, y_test = train_test_split(features, class_data, test_size=0.33, random_state=42)
#
#knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
#knn.fit(X_train,y_train)
#
#scores = knn.score(X_test, y_test, sample_weight=None)
#print "score ",scores

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors= 2)
#result = cross_val_score(clf, features, class_data, cv =3, scoring='accuracy')
#print 'Accuracy KNN : ' + repr(result.mean() * 100)

min_max = preprocessing.MinMaxScaler()
x_scaled = min_max.fit_transform(features)
dataset = pd.DataFrame(x_scaled)
dataset = dataset.values
class_data = pd.DataFrame(class_data)
class_data = class_data.values

X_train =[]
X_test = []
y_train = []
y_test = []

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
    #presisi = (float(sm.precision_score(model_result, y_test)) * 100)
    print akurasi