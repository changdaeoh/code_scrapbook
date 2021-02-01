'''
scraped from "https://m.blog.naver.com/cjh226/221359032956"
'''

import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

## 1. Load data
data = load_breast_cancer()
X = np.array(data.data)
y = np.array(data.target)
print('X.shape: {}'.format(X.shape))
print('y.shape: {}'.format(y.shape))

## 2. Build estimators
kernel = 'linear'
dual = False if X.shape[0] > X.shape[1] else True

estimator_names = ['SVC', 'LinearSVC', 'Bagging+SVC', 'Bagging+LinearSVC', 'Bagging+SVC+Multiprocess', 'Bagging+LinearSVC+Multiprocess']
estimators = [SVC(kernel=kernel),
			  LinearSVC(dual=dual),
			  BaggingClassifier(SVC(kernel=kernel), n_estimators=10, max_samples=0.1, n_jobs=1),
			  BaggingClassifier(LinearSVC(dual=dual), n_estimators=10, max_samples=0.1, n_jobs=1),
			  BaggingClassifier(SVC(kernel=kernel), n_estimators=10, max_samples=0.1, n_jobs=10),
			  BaggingClassifier(LinearSVC(dual=dual), n_estimators=10, max_samples=0.1, n_jobs=10)]

## 3. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## 4. Fit and Prediction
print('Estimator\tAccuracy\tTime')
for name, estimator in zip(estimator_names, estimators):
	start_time = time.time()
	estimator.fit(X_train, y_train)
	score = estimator.score(X_test, y_test) # accuracy
	end_time = time.time()
	print('{}\t{:.3f}\t{:.3f}'.format(name, score, (end_time-start_time)))