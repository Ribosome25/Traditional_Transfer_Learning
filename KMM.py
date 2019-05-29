# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:09:46 2019

@author: Ruibzhan

(pip install cvxopt)
"""

import numpy as np 
import math
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import check_pairwise_arrays

#%matplotlib inline

# an implementation of Kernel Mean Matchin
# referenres:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
class kmm:
    
    def __init__(self,X_target,X_source):
        X_target = np.atleast_1d(X_target)
        X_source = np.atleast_1d(X_source)
        if (X_target.shape[0] == X_target.size):
            X_target = X_target.reshape(-1,1)
        if X_source.shape[0] == X_source.size:
            X_source = X_source.reshape(-1,1)
        X_target, X_source = check_pairwise_arrays(X_target,X_source,dtype = float)
        
        self.X_target = X_target
        self.X_source = X_source
        return None
    
    def fit_kernel_mean_matching(self, kern='lin', sigma = 1, B = 1.0, eps=None, lmbd = 1):
        _X_target = self.X_target
        _X_source = self.X_source
        _coef = self._get_coef(_X_target, _X_source, kern, sigma, B, eps, lmbd)
        self.coef = _coef
        return self
    
    def _get_coef(self, X_target, X_source, kern, sigma, B, eps, lmbd):

        n_t = X_target.shape[0]
        n_s = X_source.shape[0]
        if eps == None:
            eps = B/math.sqrt(n_s)
        if kern == 'lin':
            K = np.dot(X_source, X_source.T)
            kappa = np.sum(np.dot(X_source, X_target.T)*float(n_s)/float(n_t),axis=1)
        elif kern == 'rbf':
            K = self._compute_rbf(X_source,X_source,sigma = sigma)
            kappa = np.sum(self._compute_rbf(X_source,X_target),axis=1)*float(n_s)/float(n_t)
        else:
            raise ValueError('unknown kernel')
        K = K+lmbd*np.eye(n_s)
        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1,n_s)), -np.ones((1,n_s)), np.eye(n_s), -np.eye(n_s)])
        h = matrix(np.r_[n_s*(1+eps), n_s*(eps-1), B*np.ones((n_s,)), np.zeros((n_s,))])
        
        sol = solvers.qp(K, -kappa, G, h)
        coef = np.array(sol['x'])
        
        return coef

    def _compute_rbf(X, Z, sigma=1.0):
        K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
        for i, vx in enumerate(X):
            K[i,:] = np.exp(-np.sum((vx-Z)**2, axis=1)/(2.0*sigma))
        return K
    
    def fit_featurewise_kernel_mean_matching(self, kern = 'rbf', sigma = 1, B = 1.0, eps = None, lmbd = 1):
        _coefs = np.zeros(self.X_source.shape)
        for ii in range(self.X_source.shape[1]):
            _Xs = self.X_source[:,ii].reshape(-1,1)
            _Xt = self.X_target[:,ii].reshape(-1,1)
            _each_coef = self._get_coef(_Xt, _Xs, kern, sigma, B, eps, lmbd).ravel()
            _coefs[:,ii] = _each_coef
            
        self.coef = _coefs
        
        return self



