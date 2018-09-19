# -*- coding: utf-8 -*-
"""


@author: inmkumar10
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
df = pd.read_csv('sms.csv',header='infer')
df.columns
df.head(5)
"""
   label                                            message
0      0  Go until jurong point, crazy.. Available only ...
1      0                      Ok lar... Joking wif u oni...
2      1  Free entry in 2 a wkly comp to win FA Cup fina...
3      0  U dun say so early hor... U c already then say...
4      0  Nah I don't think he goes to usf, he lives aro...
"""
"""
Here we are going to predict whether a msg is spam or not
TfidfVectorizer is used for vectorizing a sentence.
It will remove the stop word, repeated words
creating Bag of Words
We call vectorization the general process of turning a collection of text documents into 
numerical feature vectors. This specific strategy (tokenization, counting and normalization)
is called the Bag of Words or “Bag of n-grams” representation.Documents are described by
word occurrences while completely ignoring the relative position information of the words in the document.
"""
vectors = TfidfVectorizer()
Xtrain,Xtest,ytrain,ytest = train_test_split(df['message'],df['label'],random_state=11)
XtrainConverted = vectors.fit_transform(Xtrain)
XtestConverted = vectors.transform(Xtest)
clf = LogisticRegression()
clf.fit(XtrainConverted,ytrain)
#pred = clf.predict(XtestConverted)
#presision = precision_score(XtrainConverted,)
"""
Here we are calculating Accuracy,Precision,Recall,f1score
What is confusion matrix and scores
https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
"""
print('##### Accuracy #####')
accuracy = cross_val_score(clf,XtrainConverted,ytrain,cv=5) # its numpy array has 5 scores with different datasect
print(np.mean(accuracy),accuracy)
print('##### Precision #####')
presision = cross_val_score(clf,XtrainConverted,ytrain,cv=5,scoring='precision') # its numpy array has 5 scores with different datasect
print(np.mean(presision),presision)
print('##### Recall #####')
recall = cross_val_score(clf,XtrainConverted,ytrain,cv=5,scoring='recall') # its numpy array has 5 scores with different datasect
print(np.mean(recall),recall)
print('##### F1 Score #####')
f1score = cross_val_score(clf,XtrainConverted,ytrain,cv=5,scoring='f1')
print(np.mean(f1score),f1score)


from matplotlib import pyplot as plt
ypred = clf.predict_proba(XtestConverted)
false_positive_rate, recall, thresholds = roc_curve(ytest, ypred[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

"""
Scores after prediction, for testing purpose i have written the below code

"""
prediction = clf.predict(XtestConverted)
pres = precision_score(ytest,prediction,average='weighted')
recall_s= recall_score(ytest,prediction,average='weighted')
f1s = f1_score(ytest,prediction,average='weighted')
conMatrix = confusion_matrix(ytest,prediction)
