import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('titanic/train.csv')

dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

# Filling Embarked Column
dataset.isnull().sum()
dataset.Embarked.value_counts()
dataset['Embarked'] = dataset["Embarked"].fillna('S')

sns.barplot('Embarked', 'Survived', data=dataset)
plt.show()

# Filling Age column

dataset['Age'].isnull().sum()
# Using mean
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

sns.barplot('Sex', 'Survived', data=dataset)  # No. of females survived is more
plt.show()

# Encodeing Labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder = LabelEncoder()
dataset['Sex'] = labelEncoder.fit_transform(dataset['Sex'])
dataset['Embarked'] = labelEncoder.fit_transform(dataset['Embarked'])

dataset = dataset.values
one_hot_encoder = OneHotEncoder(categorical_features=[1, 7])
dataset = one_hot_encoder.fit_transform(dataset).toarray()

# Applying Algos
x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 1], test_size=0.2)

from sklearn.svm import SVC

classifier = SVC(kernel='poly', random_state=0)
classifier.fit(x_train, y_train)

yPred_svm = classifier.predict(x_test)
acc_sr = accuracy_score(yPred_svm, y_test)

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








