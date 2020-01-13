import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

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
one_hot_encoder = OneHotEncoder(categorical_features=[1,7])
dataset = one_hot_encoder.fit_transform(dataset).toarray()

# Applying Algos







