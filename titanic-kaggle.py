import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy import savetxt

dataset = pd.read_csv('titanic/train.csv')

dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

# Filling Embarked Column
dataset.isnull().sum()
dataset.Embarked.value_counts()
dataset['Embarked'] = dataset["Embarked"].fillna('S')

sns.barplot('Embarked', 'Survived', data=dataset)
plt.show()

# Filling Age column
age_train = dataset
age_apply = dataset
c = dataset['Age'].isnull().values
for i in range(len(c)):
    if c[i]:
        age_train = age_train.drop(i, axis=0)
    else:
        age_apply = age_apply.drop(i, axis=0)

age_trainY = age_train['Age']
age_trainX = age_train.drop('Age', axis=1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



labelencoder_X = LabelEncoder()
age_trainX['Sex'] = labelencoder_X.fit_transform(age_trainX['Sex'])
age_trainX['Embarked'] = labelencoder_X.fit_transform(age_trainX['Embarked'])
onehotencoder = OneHotEncoder(categorical_features=['Sex', 'Embarked'])
age_trainX = onehotencoder.fit_transform(age_trainX).toarray()

age_trainY[:, 2] = labelencoder_X.fit_transform(age_trainY[:, 2])
age_trainY[:, 6] = labelencoder_X.fit_transform(age_trainY[:, 6])
onehotencoder = OneHotEncoder(categorical_features=[1, 6])
age_cal_test = onehotencoder.fit_transform(age_trainY).toarray()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(age_trainX, age_trainY)


sns.barplot('Sex', 'Survived', data=dataset)  # No. of females survived is more
plt.show()

# Encodeing Labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder = LabelEncoder()
dataset['Sex'] = labelEncoder.fit_transform(dataset['Sex'])
dataset['Embarked'] = labelEncoder.fit_transform(dataset['Embarked'])

dataset = dataset.values
one_hot_encoder = OneHotEncoder(categorical_features=[1,7])
dataset = one_hot_encoder.fit_transform(dataset).toarray()

# Applying Algos
x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 1], test_size=0.3, random_state=4)

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(x_train, y_train)

yPred_svm = classifier.predict(x_test)
acc_sr = accuracy_score(yPred_svm, y_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
prediction_lr=model.predict(x_test)
acc_sr_lr = accuracy_score(prediction_lr, y_test)

# Test Dataset and Processing

testDataset = pd.read_csv('titanic/test.csv')
testDataset = testDataset.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)
testDataset['Age'] = testDataset['Age'].fillna(testDataset['Age'].mean())
testDataset['Fare'] = testDataset['Fare'].fillna(testDataset['Fare'].mean())

labelEncoder2 = LabelEncoder()
testDataset['Sex'] = labelEncoder2.fit_transform(testDataset['Sex'])
testDataset['Embarked'] = labelEncoder2.fit_transform(testDataset['Embarked'])

testDataset = testDataset.values
ohe = OneHotEncoder(categorical_features=[0, 6])
testDataset = ohe.fit_transform(testDataset).toarray()

x = dataset[:, 1:]
y = dataset[:,0]

svm = SVC(kernel='poly', random_state=0)
svm.fit(x, y)

yPred = svm.predict(testDataset)

submission = pd.DataFrame({'PassengerId':pd.read_csv('titanic/test.csv')['PassengerId'],'Survived':yPred})
submission.head()

filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)






