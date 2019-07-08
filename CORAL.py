# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:42:14 2019

@author: Ruibzhan
"""

import Datasets
import numpy as np
#from sklearn.preprocessing import scale
import scipy.linalg

def Coral(Xs,Xt,lmbd = 1):
    Xs = Xs-np.mean(Xs,axis = 0)
    Xt = Xt-np.mean(Xt,axis = 0)
    Cov_S = np.cov(Xs,rowvar = False) + lmbd*np.eye(Xs.shape[1])
    Cov_T = np.cov(Xt,rowvar = False) + lmbd*np.eye(Xt.shape[1])
    A_coral = np.dot(scipy.linalg.fractional_matrix_power(Cov_S, -0.5),
                     scipy.linalg.fractional_matrix_power(Cov_T, 0.5))
    Xs_new = np.dot(Xs,A_coral)

    return Xs_new, Xt
