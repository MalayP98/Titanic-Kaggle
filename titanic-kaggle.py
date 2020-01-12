import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
dataset['Age'] = dataset['Age'].fillna(29)

sns.barplot('Sex', 'Survived', data=dataset)  # No. of females survived is more
plt.show()




