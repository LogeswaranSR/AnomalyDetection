# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:31:36 2023

@author: Loges
"""

import numpy as np

class AnomalyDetectionModel:
    def __init__(self):
        self.mu=None
        self.var=None
        
    def estimate_gaussian(self, X):
        m, n = X.shape
        self.mu = np.sum(X, axis=0) / m
        self.var = np.sum((X-self.mu)**2, axis=0) / m
        return self.mu, self.var
    
    def multivariate_gaussian(self, X, mu=None, var=None):
        if(mu==None):
            mu=self.mu
        if(var==None):
            var=self.var
        coeff = np.sqrt(2*np.pi*var)
        power = ((X-mu)**2)/(2*var)
        prob = np.exp(-power)/coeff
        return np.prod(prob, axis=1, keepdims=True)
    