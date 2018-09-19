# -*- coding: utf-8 -*-
"""


@author: muthukumar_ram
D:\Python\ML\att_faces
"""
"""
Eigenvalues and EigenVectors
    - used for reducing the dimensions
    - Dimensionality Reduction
    - Principal Component Analysis
If we have more dimension values, we can't visualize. So we have to reduce the dimensions
n dimension to k- dimension k is < n
AV-> = lambdaV->
where A is the squared matrix, V-> is the vector is equal to lambda (scalar value) V->
which means vector direction wont change but magnitude will change
lambda will not be zero. 

how do you find the eigenvector for the below matrix
A = [[1,-2],[2,-3]] you have to solve the above equation to find the lambda then find the det of matrix
then solve the vector, but using numpy we can solve the problem in two lines
import numpy as np
eigenvalue, evector = np.linalg.eig(np.array([[1, -2], [2, -3]]))

you may get eigenvalue as a vector because (lambda - 5)(lambda+2)=0 which means lambda is 5 or -2
[5,-2] lambda1 and lambda2

Here is the implementation of PCA

"""
from os import walk, path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

"""
Dimensionality Reduction
We are reducing 4D data to 2D to visualize, we will get 3 different colors in the plot
R G B
pca = PCA(n_components=2) here we are mentioning the dimension size
"""
data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

"""
Face Recognition 
"""
X = []
y = []

for dirpath, _, filenames in walk('D:\\Python\\ML\\att_faces'):
    for filename in filenames:
        if filename[-3:] == 'pgm':
            img = Image.open(path.join(dirpath, filename)).convert('L')
            arr = np.array(img).reshape(10304).astype('float32') / 255.
            X.append(arr)
            y.append(dirpath)

X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=150)


X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print(X_train.shape)
print(X_train_reduced.shape)
classifier = LogisticRegression()
accuracies = cross_val_score(classifier, X_train_reduced, y_train)
print('Cross validation accuracy: %s' % np.mean(accuracies))
classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print(classification_report(y_test, predictions))


"""
We have 40 classes and their confusion matrix
Cross validation accuracy: 0.788322799471
                            precision    recall  f1-score   support

 D:\Python\ML\att_faces\s1       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s10       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s11       1.00      1.00      1.00         2
D:\Python\ML\att_faces\s12       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s13       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s14       0.67      1.00      0.80         2
D:\Python\ML\att_faces\s15       1.00      1.00      1.00         2
D:\Python\ML\att_faces\s16       1.00      0.50      0.67         2
D:\Python\ML\att_faces\s17       0.67      1.00      0.80         2
D:\Python\ML\att_faces\s18       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s19       1.00      0.67      0.80         6
 D:\Python\ML\att_faces\s2       1.00      1.00      1.00         2
D:\Python\ML\att_faces\s20       1.00      1.00      1.00         4
D:\Python\ML\att_faces\s21       1.00      1.00      1.00         4
D:\Python\ML\att_faces\s22       1.00      1.00      1.00         4
D:\Python\ML\att_faces\s23       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s24       1.00      1.00      1.00         2
D:\Python\ML\att_faces\s25       1.00      1.00      1.00         2
D:\Python\ML\att_faces\s26       1.00      1.00      1.00         2
D:\Python\ML\att_faces\s27       0.50      1.00      0.67         1
D:\Python\ML\att_faces\s28       1.00      1.00      1.00         2
D:\Python\ML\att_faces\s29       1.00      1.00      1.00         2
 D:\Python\ML\att_faces\s3       0.83      1.00      0.91         5
D:\Python\ML\att_faces\s30       1.00      1.00      1.00         5
D:\Python\ML\att_faces\s31       1.00      1.00      1.00         3
D:\Python\ML\att_faces\s33       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s34       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s35       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s36       1.00      1.00      1.00         3
D:\Python\ML\att_faces\s37       1.00      1.00      1.00         1
D:\Python\ML\att_faces\s38       1.00      0.75      0.86         4
D:\Python\ML\att_faces\s39       1.00      1.00      1.00         3
 D:\Python\ML\att_faces\s4       1.00      0.80      0.89         5
D:\Python\ML\att_faces\s40       1.00      0.50      0.67         4
 D:\Python\ML\att_faces\s5       0.60      1.00      0.75         3
 D:\Python\ML\att_faces\s6       1.00      1.00      1.00         2
 D:\Python\ML\att_faces\s7       1.00      1.00      1.00         4
 D:\Python\ML\att_faces\s8       0.83      1.00      0.91         5
 D:\Python\ML\att_faces\s9       1.00      1.00      1.00         3

               avg / total       0.95      0.93      0.93       100
""""



