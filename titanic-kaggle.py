import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('titanic/train.csv')


def view_data(x, y):
    for i in range(len(x)):
        print("{}---{} \n".format(x[i], y[i]))


# Cleaning the data

dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

emb = {}  # 'S' appeared most of the times
for i in dataset['Embarked']:
    if i in emb.keys():
        emb[i] += 1
    else:
        emb[i] = 1

c = dataset['Embarked'].isnull().values
for i in range(len(c)):
    if c[i]:
        dataset['Embarked'][i] = 'S'

sns.barplot('Pclass', 'Survived', data=dataset)

age_cal_train = dataset
age_cal_test = dataset

c = age_cal_train['Age'].isnull().values
for i in range(len(c)):
    if c[i]:
        age_cal_train = age_cal_train.drop(i, axis=0)
    else:
        age_cal_test = age_cal_test.drop(i, axis=0)

target = age_cal_train['Age'].values
age_cal_train = age_cal_train.drop('Age', axis=1).values
age_cal_test = age_cal_test.drop('Age', axis=1).values

# Regeression to calculate missing ages

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
age_cal_train[:, 2] = labelencoder_X.fit_transform(age_cal_train[:, 2])
age_cal_train[:, 6] = labelencoder_X.fit_transform(age_cal_train[:, 6])
onehotencoder = OneHotEncoder(categorical_features=[1, 6])
age_cal_train = onehotencoder.fit_transform(age_cal_train).toarray()

age_cal_test[:, 2] = labelencoder_X.fit_transform(age_cal_test[:, 2])
age_cal_test[:, 6] = labelencoder_X.fit_transform(age_cal_test[:, 6])
onehotencoder = OneHotEncoder(categorical_features=[1, 6])
age_cal_test = onehotencoder.fit_transform(age_cal_test).toarray()

from sklearn.linear_model import LinearRegression

linearReg = LinearRegression()
linearReg.fit(age_cal_train[:600], target[:600])

# Accuracy
from sklearn import metrics

y = linearReg.predict(age_cal_train[:114])
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(target[:114], y)))  # error of 12.14

linearReg.fit(age_cal_train, target)
yPred = linearReg.predict(age_cal_test)

c = dataset['Age'].isnull().values
counter = -1

for i in range(len(dataset)):
    if c[i]:
        counter += 1
        dataset['Age'][i] = int(yPred[counter])

x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

yTest = pd.read_csv('titanic/test.csv')

counter = 0
sum_age = 0
for i in range(len(x)):
    if x[i, 2] > 0:
        sum_age += x[i, 2]
    else:
        counter += 1
        x[i, 2] = int(sum_age / i + 1)
        sum_age += x[i, 2]

view_data(x, y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEn = LabelEncoder()
x[:, 0] = labelEn.fit_transform(x[:, 0])
x[:, 1] = labelEn.fit_transform(x[:, 1])
x[:, 6] = labelEn.fit_transform(x[:, 6])
ohe = OneHotEncoder(categorical_features=[0, 6])
x = ohe.fit_transform(x).toarray()

for i in [7, 8, 9, 10]:
    mini = min(x[:, i])
    maxi = max(x[:, i])
    x[:, i] = (x[:, i] - mini) / (maxi - mini)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#######################################################

# Support Vector Machine

from sklearn.svm import SVC  # accuracy = 81.**

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

yPred_svm = classifier.predict(x_test)
acc_sr = accuracy_score(yPred_svm, y_test)

########################################################

# Random Forest

from sklearn.ensemble import RandomForestClassifier  # accuracy = 76.** - 77.**

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

yPred_rfc = rfc.predict(x_test)
acc_rfc = accuracy_score(yPred_rfc, y_test)

#######################################################
# Artificial Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()

ann.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
ann.add(Dense(output_dim=6, init='uniform', activation='relu'))
ann.add(Dense(output_dim=6, init='uniform', activation='relu'))
ann.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(x_train, y_train, batch_size=30, nb_epoch=100)
yPred_ann = ann.predict(x[600:])

for i in range(len(yPred_ann)):
    if yPred_ann[i] > 0.5:
        yPred_ann[i] = 1
    else:
        yPred_ann[i] = 0

####################################################

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()
logReg.fit(x_train, y_train)

yPred_log = logReg.predict(x_test)
acc_lg = accuracy_score(yPred_log, y_test)

####################################################

# Ensemble of RandomForest, SVM, ANN

l = []
zeros = 0
ones = 0

for i in range(291):
    output = [yPred_svm[i], int(yPred_ann[i]), yPred_rfc[i]]
    for i in output:
        if i == 0:
            zeros += 1
        else:
            ones += 1
    if zeros > ones:
        l.append(0)
    else:
        l.append(1)
    zeros = 0
    ones = 0

########################################################

from sklearn.ensemble import VotingClassifier

estimator = [('svm', classifier), ('rfc', rfc), ('log_reg', logReg)]
ensemble = VotingClassifier(estimator, voting='hard')

ensemble.fit(x_train, y_train)
ensemble.score(x_test, y_test)

#########################################################

test = pd.read_csv('titanic/test.csv')
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

labelEn2 = LabelEncoder()
test['Sex'] = labelEn2.fit_transform(test['Sex'])
test['Embarked'] = labelEn2.fit_transform(test['Embarked'])

a = test['Age'].isnull()
age_train = test
age_test = test
for i in range(len(a)):
    if a[i]:
        age_train = age_train.drop(i, axis=0)
    else:
        age_test = age_test.drop(i, axis=0)

age_trainX = age_train.drop('Age', axis=1)
age_trainY = age_train['Age']
age_trainX = age_trainX.values
age_trainY = age_trainY.values

ohe2 = OneHotEncoder(categorical_features=[0, 5])
age_trainX = ohe2.fit_transform(age_trainX).toarray()

for i in [7, 8, 9]:
    print(i)
    mini = min(age_trainX[:, i])
    maxi = max(age_trainX[:, i])
    print(mini, maxi)
    age_trainX[:, i] = (age_trainX[:, i] - mini) / (maxi - mini)

reg2 = LinearRegression()
reg2.fit(age_trainX, age_trainY)

age_test = age_test.drop('Age', axis=1)
age_test = age_test.values
ohe2 = OneHotEncoder(categorical_features=[0, 5])
age_test = ohe2.fit_transform(age_test).toarray()

for i in [7, 8, 9]:
    print(i)
    mini = min(age_test[:, i])
    maxi = max(age_test[:, i])
    print(mini, maxi)
    age_test[:, i] = (age_test[:, i] - mini) / (maxi - mini)

age_test_pred = reg2.predict(age_test)

counter = -1
age_isnull = test.isnull()
for i in range(len(test)):
    if age_isnull[i]:
        counter += 1
        test['Age'][i] = age_test_pred[counter]































from sklearn.ensemble import RandomForestClassifier  # accuracy = 76.** - 77.**

rfc = RandomForestClassifier()
rfc.fit(x, y)

yPred_rfc = rfc.predict()
acc_rfc = accuracy_score(yPred_rfc, y_test)












