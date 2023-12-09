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
        if(mu==None):
            mu=self.mu
        if(var==None):
            var=self.var
        coeff = np.sqrt(2*np.pi*var)
        power = ((X-mu)**2)/(2*var)
        prob = np.exp(-power)/coeff
        return np.prod(prob, axis=1, keepdims=True)
    