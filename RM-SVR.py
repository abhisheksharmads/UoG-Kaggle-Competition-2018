import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Load the datasets which are to be trained
X_train = np.loadtxt('X_train_reg.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test_reg.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train_reg.csv', delimiter=',', skiprows=1)[:, 1]

# Start data processing over a pipline 
pipe = make_pipeline(preprocessing.StandardScaler(),SVR())

# Declare parameter to fit
hyperparameters = { 'svr__C' : range(25,40,1),'svr__kernel' : ['linear']}

# Fit model using cross-validation pipeline
cls = GridSearchCV(pipe, hyperparameters)
cls.fit(X_train, y_train)

# Predict
y_predict = cls.predict(X_test)

# Store answer into a csv file with the predicted model
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_predict_pp = np.ones((n_points, 2))
y_predict_pp[:, 0] = range(n_points)
y_predict_pp[:, 1] = y_predict
np.savetxt('regr_svr_submission.csv', y_predict_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
print(y_predict)


## Test Accuracy
features_train, features_test, target_train, target_test = train_test_split(X_train,y_train,random_state = 11)

svc2 = SVC()
svc2.fit(X_train, y_train)

y_pred2 = svc2.predict(features_test)
x_pred = svc2.predict(X_train)
acc_score3 = accuracy_score(y_train, x_pred)
acc_score2 = accuracy_score(target_test, y_pred2)

print('SVM Test Accuracy:  ', acc_score2)
print('SVM Train Accuracy:  ', acc_score3)


