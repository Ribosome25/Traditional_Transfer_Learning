# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:17:33 2019

@author: Ruibzhan
"""
import warnings

import numpy as np

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model.ridge import _solve_cholesky_kernel
import sklearn.utils

#================================
class InstanceKRR:
    """
    Attributes
    ----------
    dual_coef_ : array, shape = [n_samples] or [n_samples, n_targets]
        Representation of weight vector(s) in kernel space
    X_fit_ : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training data, which is also required for prediction. If
        kernel == "precomputed" this is instead the precomputed
        training matrix, shape = [n_samples, n_samples].
    """
    
    def __init__(self, lmbd = 1, kernel = 'rbf',gamma=None, degree=3, coef0=1,kernel_params=None):
        self.lmbd = lmbd
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
    
    def _get_kernel(self, X, Y = None, indicted_gamma = None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {'gamma': self.gamma, 
                      'degree': self.degree,
                      'coef0': self.coef0}
        if indicted_gamma: # if not indicated, use self.gamma
            params['gamma'] = indicted_gamma
        return pairwise_kernels(X,Y,metric = self.kernel,filter_params = True, **params)
    
    def fit(self, X, y = None, sample_weight = None):
        X,y = sklearn.utils.check_X_y(X,y, accept_sparse = ('csr', "csc"), multi_output=True,
                                      y_numeric=True)
        n_samples, n_features = X.shape
        
        # sample weights
        K = self._get_kernel(X)
        lmbd = np.atleast_1d(self.lmbd)# why??
        one_alpha = (lmbd == lmbd[0]).all()
        
        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
            ravel = True
        
        if one_alpha:
            K.flat[::n_samples+1] += lmbd[0] # Save memory
            try:
                self.dual_coef_ = np.linalg.solve(K,y)
            except np.linalg.LinAlgError:
                warnings.warn("Singular matrix in solving dual problem. Using least-squares solution instead.")
                self.dual_coef_ = np.linalg.lstsq(K,y)[0]
            K.flat[::n_samples+1] -= lmbd[0]
            
        else:
            warnings.warn("Multiple lambda values, check in later")
            
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()
        
        self.X_fit_ = X
        self.y_fit_ = y
        self.n_source_sampels = n_samples
        
        return self
    
    def kRR_predict(self,x):
        sklearn.utils.validation.check_is_fitted(self,["X_fit_","y_fit_","dual_coef_"])
        k = self._get_kernel(x,self.X_fit_)
        y = np.dot(k,self.dual_coef_) 
        return y
    
    def _get_D_matrix(self, Xaux, yaux, eta = 1):
        # a is the self.dual_coef
        Xy_source = np.hstack((self.X_fit_,self.y_fit_))
        Xy_aux = np.hstack((Xaux,yaux))
        self.E_mat = self._get_kernel(Xy_source,Xy_aux,indicted_gamma = eta)
        K_SAX = self._get_kernel(self.X_fit_,Xaux)
        Stacking = np.dot(self.dual_coef_.reshape(1,-1),K_SAX)
        Stacked = np.tile(Stacking,(self.n_source_sampels,1))
        self.D_mat = np.multiply(self.E_mat,Stacked)
        
        return self
        
    def Solve_alpha(self, Xaux, Yaux, eta = 1, lmbd2 = 1):
        sklearn.utils.validation.check_is_fitted(self,["X_fit_","dual_coef_","y_fit_"])
        Xaux,Yaux = sklearn.utils.check_X_y(Xaux,Yaux, accept_sparse = ('csr', "csc"), multi_output=True,
                                      y_numeric=True)
        if len(Yaux.shape) == 1:
            Yaux = Yaux.reshape(-1,1)
        
        self._get_D_matrix(Xaux, Yaux, eta = eta)
        D_mat = self.D_mat
        n_samples, n_features = D_mat.shape
        
        DD_mat = np.dot(D_mat, D_mat.T)
        #=== cvx.qp
        # J = ||y-D'*alphas||^2 + lmbd/2 * ||alphas||2
        DD_mat.flat[::n_samples+1] += lmbd2/2
        P = matrix(DD_mat)
        Q = matrix(np.dot(D_mat,Yaux)) # q is given in q'
        G = matrix(np.diag(-1*np.ones(n_samples)))
        h = matrix(np.zeros(n_samples).reshape(-1,1))
        self.alphas = np.array(solvers.qp(P,Q,G,h)['x'])
      
        self.n_aux_samples = Xaux.shape[0]
        self.X_aux_ = Xaux
        self.Y_aux_ = Yaux
        
        return self
    
    def predict(self,X):
        Weights_S = np.multiply(self.alphas.ravel(),np.sum(self.E_mat,axis = 1))
        Weights_all = np.hstack((np.ones(self.n_aux_samples) ,(Weights_S/np.max(Weights_S))))
        Ysp = np.vstack((self.Y_aux_, self.y_fit_))
        Xsp = np.vstack((self.X_aux_, self.X_fit_))
        KK = self._get_kernel(Xsp)
        Kx = self._get_kernel(Xsp,X)
        
        self.a_prediction = _solve_cholesky_kernel(KK,Ysp,self.lmbd,sample_weight=Weights_all)
        
        y_prediction = np.dot(Kx.T,self.a_prediction)
        
        return y_prediction
    
