#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:18:38 2024

@author: daniel_long
"""

#libararies used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

#read in data
df = pd.read_csv('encoded_data.csv')
#transform placement to top 4 and bottom 4
df['placement'] = np.where(df['placement'] <= 4, 1, 0)


#target and data seperation
y = df['placement']
x = df.drop('placement', axis = 1)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

# Drop the first 5 columns for binary transformation
x_train_binary = x_train.iloc[:, 5:].copy()
x_test_binary = x_test.iloc[:, 5:].copy()

#data for gaussian using non-binary data
x_train_gaussian = x_train.iloc[:, :5].copy()
x_test_gaussian = x_test.iloc[:, :5].copy()

# Convert remaining features to binary (0 or 1)
x_train_binary[x_train_binary > 0] = 1
x_test_binary[x_test_binary > 0] = 1

#saving train and test datasets for replication.
train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1)
train_data_binary = pd.concat([x_train_binary, y_train], axis=1)
test_data_binary = pd.concat([x_test_binary, y_test], axis=1)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
x_train_gaussian.to_csv('train_data_gaussian.csv', index=False)
x_test_gaussian.to_csv('test_data_gaussian.csv', index=False)
x_train_binary.to_csv('train_data_binary.csv', index=False)
x_test_binary.to_csv('test_data_binary.csv', index=False)

#Multinomial Naive Bayes
mnb = MultinomialNB().fit(x_train, y_train)

predictions_mnb = mnb.predict(x_test)

#accuracies for MNB
accuracy_mnb = accuracy_score(y_test, predictions_mnb)
print(f"Multinomial Naive Bayes Accuracy: {accuracy_mnb:.2f}")

#confusion matrix nb
cm_mnb = confusion_matrix(y_test, predictions_mnb, labels = mnb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm_mnb, display_labels = mnb.classes_)
disp.plot(cmap = 'cividis')
plt.title("Multinomial Naive Bayes Confusion Matrix")
plt.show()

#Gaussian Naive Bayes
gnb = GaussianNB().fit(x_train_gaussian, y_train)
predictions_gnb = gnb.predict(x_test_gaussian)

#accuracy GNB 
accuracy_gnb = accuracy_score(y_test, predictions_gnb)
print(f"Gaussian Naive Bayes Accuracy: {accuracy_gnb:.2f}")

#conrusion matrix GNB
cm_gnb = confusion_matrix(y_test, predictions_gnb, labels=gnb.classes_)
disp_gnb = ConfusionMatrixDisplay(confusion_matrix=cm_gnb, display_labels=gnb.classes_)
disp_gnb.plot(cmap='cividis')
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.show()

# Bernoulli Naive Bayes
bnb = BernoulliNB().fit(x_train_binary, y_train)
predictions_bnb = bnb.predict(x_test_binary)

#accuracy BNB
accuracy_bnb = accuracy_score(y_test, predictions_bnb)
print(f"Bernoulli Naive Bayes Accuracy: {accuracy_bnb:.2f}")

#confusion matrix BNB
cm_bnb = confusion_matrix(y_test, predictions_bnb, labels=bnb.classes_)
disp_bnb = ConfusionMatrixDisplay(confusion_matrix=cm_bnb, display_labels=bnb.classes_)
disp_bnb.plot(cmap='cividis')
plt.title("Bernoulli Naive Bayes Confusion Matrix")
plt.show()