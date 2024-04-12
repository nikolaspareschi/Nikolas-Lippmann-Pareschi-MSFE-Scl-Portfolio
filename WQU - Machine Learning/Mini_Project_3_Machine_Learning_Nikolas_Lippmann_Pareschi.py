# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:13:04 2018

@author: Nikolas
"""

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split



# Importing the data files provided by  WQU
df_train = pd.read_csv("train__titanic.csv")
df_test = pd.read_csv("test_titanic.csv")

# Target variable

survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

data.info()

# Dealing with Missing Value

data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Dealing with categorical variables:

data.info()

data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()

# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()

data.info()

data_train = data.iloc[:891]
data_test = data.iloc[891:]

X = data_train.values
test = data_test.values
y = survived_train.values

# Fittin the data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#            splitter='best')

# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)

#df_test[['PassengerId', 'Survived']].to_csv('1st_dec_tree.csv', index=False)


print 'Our test accuracy is', clf.score(X_test, y_test)

#
#test_accuracy_3 = np.empty(len(dep))
#test_accuracy_3[i] = clf.score(X_test, y_test)

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.33, random_state=42, stratify=y)



# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a Decision Tree Classifier
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()