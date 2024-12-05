# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:41:07 2024

@author: Daniel Long
"""
#needed library
import numpy as np

#class for the decison stump
class Stump:
    def __init__(self):
        #which way we are looking 
        self.decision_side = 1
        #index of currently looked at feature
        self.feature_index = None
        #seperation threshold for split
        self.threshold = None
        
    #define predicting points   
    def predict(self, X):
        
        n_samples = X.shape[0]
        #look at only one feature at a time
        X_column = X[:,self.feature_index]
        
        #initially set all predictions to 1
        predictions = np.ones(n_samples)
        #set predictions to 1 if they are below threshold or overthreshold in decision_side -1
        if self.decision_side == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
            
        return predictions
        
class AdaBoost:
    #default number of classifiers is set to 5
    def __init__(self, n_clf = 5):
        self.n_clf = n_clf
    #define training the AdaBoost
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        #initial weights STEP 1 of ALGO
        w = np.full(n_samples, (1/n_samples))
        
        #empty list for weak learners to be stored
        self.clfs = []
        #loop through number of weak learners STEP 2 of ALGO
        for _ in range(self.n_clf):
            #weak learner is as stump
            clf = Stump()
            
            #set initial error to beat
            min_error = float('inf')
            
            #loop through each feature STEP 3 of ALGO
            for feature_i in range(n_features):
                #loop only at the one feature
                X_column = X[:,feature_i]
                #get different possible thresholds
                thresholds = np.unique(X_column)
                #loop through each possible separation
                for threshold in thresholds:
                    #set decision_side back to 1
                    p = 1
                    #set initial predictions to 1
                    predictions = np.ones(n_samples)
                    #make predictions based on threshold
                    predictions[X_column < threshold] = -1
                    
                    #get weighted error based on current weights STEP 4 of ALGO
                    missclassified = w[y != predictions]
                    error = sum(missclassified)
                    
                    #if error is worse than 0.5 we can flip the guesses and get better error (flip decision_side)
                    if error > 0.5:
                        error = 1-error
                        p = -1
                    #if the error is less than what we have seen assign this as the seperating feature
                    if error < min_error:
                        min_error = error
                        clf.decision_side = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        
            #so we do not divide by 0 in the case of no error
            EPS = 1e-10  
            #calculate error  STEP 5 of ALGO
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            
            #predictions for weight calculation
            predictions = clf.predict(X)
            
            #set new weights STEP 6 of ALGO
            w *= np.exp(-clf.alpha * y * predictions)
            
            #normalize the new weights STEP 7 of ALGO
            w /= np.sum(w)
            
            #save the weak learner 
            self.clfs.append(clf)
            
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis = 0)
        y_pred = np.sign(y_pred)
        return y_pred
            
        
        