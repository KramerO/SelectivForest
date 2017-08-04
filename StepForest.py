# -*- coding: utf-8 -*-
"""
@author: Oliver
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, KFold
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

class SelectedRandomForest(object):
     
    
    """
    kontrucktor setzen
    """
     def __init__(self, Type, n_estimators, n_jobs, n_features,inner):
          #self.X = X
          #self.y = y
          self.Type = Type
          self.n_estimators = n_estimators
          self.n_jobs = n_jobs
          self.n_features = n_features
          self.inner = inner
          
          
     """
     fit funktion
     
     kleine Forestss für die feature selection
     große Forest für die vorhersage
     """
     def fit(self,Samples,Target):          
          if self.Type == "Regression":
               X_train, X_test, y_train, y_test = train_test_split(Samples, Target,
                                                                   test_size=0.33)
               skf = KFold( n_splits=self.inner)
               #print(skf)
               Forest_small = RandomForestRegressor(n_estimators=self.n_estimators,
                                                    n_jobs=self.n_jobs,
                                                    verbose=False)
          if self.Type == "Classification":
               X_train, X_test, y_train, y_test = train_test_split(Samples, Target,
                                                                   test_size=0.33,
                                                                   stratify=Target)
               skf = StratifiedKFold(n_splits =5)
               Forest_small = RandomForestClassifier(n_estimators=self.n_estimators,
                                                     n_jobs=self.n_jobs,
                                                     verbose=False)
          
          Importances = []
          for train_index, test_index in skf.split(X_train,y_train):
               #print("TRAIN:", train_index, "TEST:", test_index)
               Forest_small.fit(X_train[train_index,:],y_train[train_index])
               Importances.append(Forest_small.feature_importances_) # featureeinfluss
          Importances = np.vstack(Importances)
          Mean_Impos = np.mean(Importances,axis=0) # feature einfluss relativieren
          #print (Mean_Impos)
          indices = np.argsort(Mean_Impos)[::-1][:self.n_features]
          
          if self.Type == "Regression":
               Forest_big = RandomForestRegressor(n_estimators=self.n_estimators,
                                                  n_jobs=self.n_jobs,
                                                  verbose=False)
          if self.Type == "Classification":
               Forest_big = RandomForestClassifier(n_estimators=self.n_estimators,
                                                   n_jobs=self.n_jobs,
                                                   verbose=False)
               
               
          
          Forest_big.fit(X_train[:,indices],y_train)
          self.Forest_big = Forest_big
          self.indices = indices
          
          
     def predict(self,NewData):
         #NewData = self.Forest_big.transform(NewData)
         return self.Forest_big.predict(NewData[:,self.indices])
          
     def predict_probas(self,NewData):
          #NewData = self.Forest_big.transform(NewData)
          return self.Forest_big.predict_proba(NewData[:,self.indices])
      
        
        
        
        