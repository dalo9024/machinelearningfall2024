#libraries used
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#read in data
df = pd.read_csv("encoded_data.csv")

#use only numeric predictors and convert label to top4 bottom 4
numeric_df = df[["gold_left", "level", "total_damage_to_players", "players_eliminated", "num_traits"]]
label = df['placement'].apply(lambda x: 1 if x <= 4 else 0)


#training and test split 
X_train, X_test, y_train, y_test = train_test_split(numeric_df, label, test_size=0.2, random_state=100)

#save training and testing data
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv("training_data.csv", index=False)
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv("testing_data.csv", index=False)

#different kernals to use
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

#train and evaluate the SVM with different kernels
for kernel in kernels:
    
    if kernel == 'sigmoid':
        bcoef = -5
    else:
        bcoef = 0
    
    svm_model = SVC(kernel=kernel, C = 10, degree = 5, coef0 = bcoef)
    svm_model.fit(X_train, y_train)
    print(svm_model.n_support_)
        
    #predict test values
    y_pred = svm_model.predict(X_test)
    
    #assign accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    #accuracies
    print(f"Kernel: {kernel}")
    print(f"Accuracy: {accuracy:.4f}")
    
    #plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis_r')
    plt.title(f'Confusion Matrix for {kernel} Kernel')
    plt.show()
