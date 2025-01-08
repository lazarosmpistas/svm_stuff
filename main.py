from sklearn import svm
from keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


#GLOBAL VARIABLES
TRAIN_SIZE = 50000
TEST_SIZE = 10000
CROSS_VALIDATION_SPLIT = 0.1
num_of_neuron_layers = [128, 32,  10]
epochs = 100
batch_size = 1
n = 0.001



#DATASET HANDLING
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=CROSS_VALIDATION_SPLIT, random_state=42)

class_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

X_train = X_train.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)
#X_val = X_val.reshape(-1, 32*32*3)
X_train = X_train.astype('float64') / 255
X_test = X_test.astype('float64') / 255
#X_val = X_val.astype('float64') / 255

y_test = np.reshape(y_test, TEST_SIZE)
y_train = np.reshape(y_train, TRAIN_SIZE)
#y_val = np.reshape(y_val, int(TRAIN_SIZE * CROSS_VALIDATION_SPLIT))

pca = PCA(n_components=1000)
pca.fit(X_train)
X_train = pca.transform(X_train)
#pca.fit(X_test)
X_test = pca.transform(X_test)


clf = svm.SVC(kernel='rbf', C=10, decision_function_shape='ovr', gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print(accuracy_score(y_train, y_pred))
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))