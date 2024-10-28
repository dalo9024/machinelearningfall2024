#libararies used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB

#read in data
df = pd.read_csv('encoded_data.csv')

#transform placement into top 4/bottom 4 and encode as 1 = top4
df['placement'] = np.where(df['placement']> 4, 0,1)


#target and data seperation
y = df['placement']
x = df.drop('placement', axis = 1)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

#saving train and test datasets for replication.
train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

#Logistic Regression
lr = LogisticRegression(penalty = 'l2', solver = 'newton-cg').fit(x_train, y_train)

#test_data predicitons lr
predictions_lr = lr.predict(x_test)

#confusion matrix lr
cm_lr = confusion_matrix(y_test, predictions_lr, labels = lr.classes_)
disp_lr = ConfusionMatrixDisplay(confusion_matrix = cm_lr, display_labels = lr.classes_)
disp_lr.plot(cmap = 'viridis_r')
plt.show()

#Naive Bayes
nb = MultinomialNB().fit(x_train, y_train)

#test_data predicitons nb
predictions_nb = nb.predict(x_test)

#confusion matrix nb
cm_nb = confusion_matrix(y_test, predictions_nb, labels = nb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm_nb, display_labels = nb.classes_)
disp.plot(cmap = 'coolwarm')
plt.show()




