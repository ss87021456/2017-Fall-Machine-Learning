import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('iris.csv')
X = df.drop('class',axis=1)
y = df['class']

ITERATION_TIME = 100
label = ['setosa','versicolor','virginica']


# result container
Decision_acc = 0
KNN_acc = 0
Naive_acc = 0
Decision_result = [(0,0),(0,0),(0,0)]
KNN_result = [(0,0),(0,0),(0,0)]
Naive_result = [(0,0),(0,0),(0,0)]

for _ in range(ITERATION_TIME):
	# separate for 70% training 30% testing
	X_train, X_test, y_train, y_test = train_test_split(
		     X, y, test_size=0.3)
	
	# Decision tree model
	Decision_clf = tree.DecisionTreeClassifier()
	Decision_clf = Decision_clf.fit(X_train, y_train)	
	Decision_pre = Decision_clf.predict(X_test)
	Decision_acc += round(accuracy_score(y_test, Decision_pre),4)

	result = precision_recall_fscore_support(y_test,Decision_pre)
	for idx ,element in enumerate(result[:3]):
		info_tuple = element[:2]
		Decision_result[idx] = (round(Decision_result[idx][0] + info_tuple[0]/ITERATION_TIME,4), round(Decision_result[idx][1] + info_tuple[1]/ITERATION_TIME,4))

	# KNN model K=5
	KNN_clf = KNeighborsClassifier(n_neighbors=5)
	KNN_clf.fit(X_train, y_train) 	
	KNN_pre = KNN_clf.predict(X_test)
	KNN_acc += round(accuracy_score(y_test, KNN_pre),4)
	result = precision_recall_fscore_support(y_test,KNN_pre)
	for idx ,element in enumerate(result[:3]):
		info_tuple = element[:2]
		KNN_result[idx] = (round(KNN_result[idx][0] + info_tuple[0]/ITERATION_TIME,4), round(KNN_result[idx][1] + info_tuple[1]/ITERATION_TIME,4))
	
	# Naive Bayes model K=5
	Naive_clf = GaussianNB()
	Naive_clf.fit(X_train, y_train)
	Naive_pre = Naive_clf.predict(X_test)
	Naive_acc += round(accuracy_score(y_test, Naive_pre),4)
	result = precision_recall_fscore_support(y_test,Naive_pre)
	for idx ,element in enumerate(result[:3]):
		info_tuple = element[:2]
		Naive_result[idx] = (round(Naive_result[idx][0] + info_tuple[0]/ITERATION_TIME,4), round(Naive_result[idx][1] + info_tuple[1]/ITERATION_TIME,4))

print "After {0}-time testing result:".format(ITERATION_TIME)
print "\nDecision Tree Accuracy =",Decision_acc/ITERATION_TIME
for idx, element in enumerate(Decision_result):
	print "{0}\tprecision: {1} recall {2}".format(label[idx],element[0],element[1])

print "\nK-Nearest Neighbor Accuracy =",KNN_acc/ITERATION_TIME
for idx, element in enumerate(KNN_result):
	print "{0}\tprecision: {1} recall {2}".format(label[idx],element[0],element[1])

print "\nNaive Bayes Accuracy =",Naive_acc/ITERATION_TIME
for idx, element in enumerate(Naive_result):
	print "{0}\tprecision: {1} recall {2}".format(label[idx],element[0],element[1])