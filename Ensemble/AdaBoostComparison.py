# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:11:06 2024

@author: Daniel Long
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from myAdaBoost import AdaBoost
import seaborn as sns  # For heatmap visualization

data = pd.read_csv('encoded_data.csv')

# Separate features (X) and target (y)
X = data.drop(columns=['placement'])
y = data['placement']

# Reassign y to binary labels: -1 for placement 5-8, 1 for placement 1-4
y = y.apply(lambda p: -1 if p >= 5 else 1)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size  = 0.2, random_state = 5)


# Range of estimators
n_estimators = list(range(5, 201, 10))

# Store accuracies
custom_accuracies = []
sklearn_accuracies = []

for n in n_estimators:
    # Custom AdaBoost
    custom_clf = AdaBoost(n_clf=n)
    custom_clf.fit(X_train.values, y_train.values)  # Convert DataFrame to NumPy arrays
    y_pred_custom = custom_clf.predict(X_test.values)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    custom_accuracies.append(acc_custom)

    # Scikit-learn AdaBoost
    sklearn_clf = AdaBoostClassifier(n_estimators=n, random_state=5, algorithm = 'SAMME')
    sklearn_clf.fit(X_train, y_train)
    y_pred_sklearn = sklearn_clf.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    sklearn_accuracies.append(acc_sklearn)

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.plot(n_estimators, custom_accuracies, label='Custom AdaBoost', marker='o', linestyle='-')
plt.plot(n_estimators, sklearn_accuracies, label='Scikit-learn AdaBoost', marker='s', linestyle='--')
plt.title('Accuracy vs Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

optimal_custom_n = n_estimators[np.argmax(custom_accuracies)]
optimal_sklearn_n = n_estimators[np.argmax(sklearn_accuracies)]

custom_clf = AdaBoost(n_clf=optimal_custom_n)
custom_clf.fit(X_train.values, y_train.values)  # Convert DataFrame to NumPy arrays

# Predict using custom AdaBoost
y_pred_custom = custom_clf.predict(X_test.values)

# Evaluate custom AdaBoost
acc_custom = accuracy_score(y_test, y_pred_custom)
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)

print("Custom AdaBoost")
print("Accuracy:", acc_custom)

# Visualize custom AdaBoost confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_custom, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 1], yticklabels=[-1, 1])
plt.title("Confusion Matrix: Custom AdaBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Use scikit-learn's AdaBoostClassifier for comparison
sklearn_clf = AdaBoostClassifier(n_estimators= optimal_sklearn_n, random_state=5, algorithm = 'SAMME')
sklearn_clf.fit(X_train, y_train)

# Predict using sklearn AdaBoost
y_pred_sklearn = sklearn_clf.predict(X_test)

# Evaluate sklearn AdaBoost
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)

print("\nScikit-learn AdaBoost")
print("Accuracy:", acc_sklearn)


# Visualize sklearn AdaBoost confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_sklearn, annot=True, fmt='d', cmap='Greens', xticklabels=[-1, 1], yticklabels=[-1, 1])
plt.title("Confusion Matrix: Scikit-learn AdaBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()