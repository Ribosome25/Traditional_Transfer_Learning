# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:14:45 2019

@author: Ruibzhan
"""

from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import scale

def fit_SDA(Xs,Xt,n_components = 30, flag = False):
    Xs = scale(Xs)
    Xt = scale(Xt)
    pca = PCA(n_components=n_components)
    pca.fit(Xs)
    pc_S = pca.components_ # n_pcs x n_fts. Transposted from matlab one 
    pe_S = pca.explained_variance_
    prj_S = pca.fit_transform(Xs)
    
    pca.fit(Xt)
    pc_T = pca.components_
    pe_T = pca.explained_variance_
    prj_T = pca.fit_transform(Xt)
    
    var_ratio = np.diag(np.multiply( np.power(pe_S,-0.5), np.power(pe_T,0.5) ))
    Ms = np.dot( np.dot( np.dot(pc_S.T,pc_S) ,pc_T.T) , var_ratio )
    Mt = pc_T.T
    
    newS = np.dot(Xs , Ms)
    newT = np.dot(Xt , Mt)
    
    if flag:
        return np.dot(prj_S, var_ratio) , prj_T
    
    return newS, newT
