#libraries used
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

#read in data
df = pd.read_csv('encoded_data.csv')
#transform placement to top 4 and bottom 4
df['placement'] = np.where(df['placement'] <= 4, 1, 0)

#target and data seperation
y = df['placement']
x = df.drop('placement', axis = 1)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

def accuracy_confusion_matrix(model, x_test, y_test, title):
    """
    Function to display a confusion matrix for a given model.
    """
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{title} Accuracy: {accuracy:.2f}")
    disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap='cividis')
    plt.title(title)
    plt.show()

# Decision Tree 1
tree1 = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=0)
tree1.fit(x_train, y_train)

#accuracy and confusion matrix tree 1
accuracy_confusion_matrix(tree1, x_test, y_test, "Decision Tree 1")

#plot of tree 1
plt.figure(figsize=(12, 8), dpi=300) 
plot_tree(tree1, feature_names=x_train.columns, filled=True, rounded=True, class_names=['Bottom 4', 'Top 4'])
plt.title('Tree 1')
plt.show()


# Decision Tree 2
tree2 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0, max_features="sqrt")
tree2.fit(x_train, y_train)

#accuracy and confusion matrix tree 2
accuracy_confusion_matrix(tree2, x_test, y_test, "Decision Tree 2")

#plot of tree 2
plt.figure(figsize=(12, 8), dpi=300) 
plot_tree(tree2, feature_names=x_train.columns, filled=True, rounded=True, class_names=['Bottom 4', 'Top 4'])
plt.title('Tree 2')
plt.show()

# Decision Tree 3
tree3 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=4, criterion='gini', random_state=100, max_features='log2')
tree3.fit(x_train, y_train)

#accuracy and confusion matrix tree 3
accuracy_confusion_matrix(tree3, x_test, y_test, "Decision Tree 3")

#plot of tree 3
plt.figure(figsize=(12, 8), dpi=300) 
plot_tree(tree3, feature_names=x_train.columns, filled=True, rounded=True, class_names=['Bottom 4', 'Top 4'])
plt.title('Tree 3')
plt.show()
