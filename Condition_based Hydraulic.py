import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import statistics
import pandas as pd

# Data loading HZ 1
TS1 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/TS1.txt")
TS2 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/TS2.txt")
TS3 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/TS3.txt")
TS4 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/TS4.txt")
VS1 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/VS1.txt")
CE = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/CE.txt")
CP = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/CP.txt")
SE = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/SE.txt")

# Data loading HZ 10
FS1 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/FS1.txt")
FS2 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/FS2.txt")

# Data loading HZ 100
PS1 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/PS1.txt")
PS2 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/PS2.txt")
PS3 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/PS3.txt")
PS4 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/PS4.txt")
PS5 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/PS5.txt")
PS6 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/PS6.txt")
EPS1 = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/EPS1.txt")

# Data concatenate
X = np.concatenate((TS1,TS2,TS3,TS4,VS1,CE,CP,SE,FS1,FS2,PS1,PS2,PS3,PS4,PS5,PS6,EPS1),axis=1)
Y = np.loadtxt("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/profile.txt")

# Save dataset
np.save("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/independentfeature.npy",X)
np.save("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/dependentvariable.npy",Y)

# Load data
X = np.load("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/independentfeature.npy")
Y = np.load("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/dependentvariable.npy")
DF_Y = pd.DataFrame(Y)
# Count the percentage of each class
per_class = DF_Y[0].value_counts(normalize=True)

# Convert integer to categorical
oe = OrdinalEncoder()
oe.fit(Y)
YY = oe.transform(Y)

# Response 1
# Ranking of the feature
model = RandomForestClassifier()
model.fit(X,YY[:,0])
importance = model.feature_importances_
res = -np.sort(-importance) # Sort with descending order
ind = np.argsort(importance)[::-1][:400]  # Index of top 400
score1 = []
for a in range(1,200,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,0][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,0][test]
        X_test = X[:,ind[:a]][test,:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score1.append(statistics.mean(sub_score))
    pyplot.plot(score1)

# Response 2
model.fit(X,YY[:,1])
importance = model.feature_importances_
res = -np.sort(-importance) # Sort with descending order
ind = np.argsort(importance)[::-1][:400]  # Index of top 400
# np.argsort(importance) index of ascending order (small to large)

score2 = []
for a in range(1,200,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,1][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,1][test]
        X_test = X[:,ind[:a]][test,:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score2.append(statistics.mean(sub_score))
    pyplot.plot(score2)

# Response 3
model.fit(X,YY[:,2])
importance = model.feature_importances_
res = -np.sort(-importance) # Sort with descending order
ind = np.argsort(importance)[::-1][:400]  # Index of top 400
score3 = []
for a in range(1,200,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,2][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,2][test]
        X_test = X[:,ind[:a]][test,:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score3.append(statistics.mean(sub_score))
    pyplot.plot(score3)

# Response 4
model.fit(X,YY[:,3])
importance = model.feature_importances_
res = -np.sort(-importance) # Sort with descending order
ind = np.argsort(importance)[::-1][:400]  # Index of top 400
score4 = []
for a in range(1,200,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,3][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,3][test]
        X_test = X[:,ind[:a]][test,:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score4.append(statistics.mean(sub_score))
    pyplot.plot(score4)

# Response 5
model.fit(X,YY[:,4])
importance = model.feature_importances_
res = -np.sort(-importance) # Sort with descending order
ind = np.argsort(importance)[::-1][:400]  # Index of top 400
score5 = []
for a in range(1,200,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,4][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,4][test]
        X_test = X[:,ind[:a]][test,:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score5.append(statistics.mean(sub_score))
    pyplot.plot(score5)

## Logistic regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = RandomForestClassifier()
model.fit(X,YY[:,3])
importance = model.feature_importances_
res = -np.sort(-importance) # Sort with descending order
ind = np.argsort(importance)[::-1][:400]  # Index of top 400
score6 = []
for a in range(1,200,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,3][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,3][test]
        X_test = X[:,ind[:a]][test,:]
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score6.append(statistics.mean(sub_score))
    pyplot.plot(score6)

## Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier

score7 = []
for a in range(1,100,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,3][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,3][test]
        X_test = X[:,ind[:a]][test,:]
        gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=2)
        gb_clf.fit(X_train, y_train)
        y_pred = gb_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score7.append(statistics.mean(sub_score))
    pyplot.plot(score7)
np.save("/Users/mengkaixu/Desktop/Dataset/UCI Dataset/data/GB_score.npy",score7)

## Support Vector Machine
from sklearn import svm

score8 = []
for a in range(1,100,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,3][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,3][test]
        X_test = X[:,ind[:a]][test,:]
        rbf_svc = svm.SVC(kernel='rbf')
        rbf_svc.fit(X_train, y_train)
        y_pred = rbf_svc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score8.append(statistics.mean(sub_score))
    pyplot.plot(score8)

## ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier
score9 = []
for a in range(1,200,1):
    kfold = KFold(5, True, 1)
    sub_score = []
    for train, test in kfold.split(X):
        y_train = YY[:,3][train]
        X_train = X[:,ind[:a]][train,:]
        y_test = YY[:,3][test]
        X_test = X[:,ind[:a]][test,:]
        extra = ExtraTreesClassifier()
        extra.fit(X_train, y_train)
        y_pred = extra.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sub_score.append(acc)
    score9.append(statistics.mean(sub_score))
    pyplot.plot(score9)