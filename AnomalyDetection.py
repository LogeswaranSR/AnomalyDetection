# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:31:36 2023

@author: Loges
"""

import numpy as np

class AnomalyDetectionModel:
    '''Class for Anomaly Detection.
    Model works based on Gaussian/Normal Distribution.
    '''
    def __init__(self):
        self.mu=None
        self.var=None
        self.epsilon=None
        
    def estimate_gaussian(self, X):
        '''
        

        Parameters
        ----------
        X : numpy.ndarray
            Dataset to be mapped to Gaussian Distribution. For shape (m, n), the distribution will be mapped for all n features.

        Returns
        -------
        mu : numpy.ndarray
            Mean of the distribution.  
        var : numpy.ndarray
            Variance of the distribution.

        '''
        m, n = X.shape
        self.mu = np.sum(X, axis=0) / m
        self.var = np.sum((X-self.mu)**2, axis=0) / m
        return self.mu, self.var
    
    def multivariate_gaussian(self, X, mu=None, var=None):
        '''
        Calculate Gaussian Probabillity Values

        Parameters
        ----------
        X : numpy.ndarray
            Data to find probabilities; Only 2D arrays accepted
        mu : numpy.ndarray, optional
            Mean of the Distribution. Already calculated in estimate_gaussian method.  The default is None.
        var : numpy.ndarray, optional
            Variance of the Distribution. Already calculated in estimate_gaussian method. The default is None.

        Returns
        -------
        prob: numpy.ndarray
            Probabilities of the given Data in the distribution.

        '''
        if(mu is None):
            mu=self.mu
        if(var is None):
            var=self.var
        coeff = np.sqrt(2*np.pi*var)
        power = ((X-mu)**2)/(2*var)
        prob = np.exp(-power)/coeff
        return np.prod(prob, axis=1)
    
    def select_threshold_value(self, anomalies, probabilities):
        best_epsilon = 0
        best_F1 = 0
        F1 = 0
        step_size = max(probabilities) - min(probabilities)
        step_size = step_size/1000
        for eps in np.arange(min(probabilities), max(probabilities), step_size):
            pred = (probabilities <= eps).astype(np.int32)
            F1 = self.f1_score(anomalies, pred)
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = eps
        self.epsilon = best_epsilon
        return best_epsilon, best_F1
    
    def fit(self, X_val, anomalies, fit_data=None):
        mu, var = self.estimate_gaussian(fit_data if fit_data is not None else X_val)
        prob = self.multivariate_gaussian(X_val, mu, var)
        eps, F1 = self.select_threshold_value(anomalies, prob)
        return F1
    
    def f1_score(self, y_true, y_pred):
        tp = np.sum((y_pred==1)&(y_true==1))
        fp = np.sum((y_pred==1)&(y_true==0))
        fn = np.sum((y_pred==0)&(y_true==1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2*precision*recall)/(precision+recall)
        return F1
    
    def predict(self, predictors:np.ndarray, prob:bool = False):
        if not prob:
            predictors = self.multivariate_gaussian(predictors)
        anomaly = (predictors < self.epsilon)
        return anomaly
    
    def evaluate(self, X, anomalies):
        pred = self.predict(X)
        score = self.f1_score(anomalies, pred)
        return score
    
    def find_anomalies(self, X):
        '''
        Finds Anomalies for the given items

        Parameters
        ----------
        X : np.ndarray
            Feature Vectors of itens to be predicted

        Returns
        -------
        anomalies_count : int
            Count of anomalies found
        anomalies : np.ndarray
            Array of indices of the anomalies in the given data

        '''
        prob = self.predict(X)
        anomalies_count = np.sum((prob).astype(int))
        anomalies = np.array([ind for ind in range(len(prob)) if prob[ind]])
        return anomalies_count, anomalies