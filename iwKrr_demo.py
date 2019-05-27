# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:36:27 2019

@author: Ruibzhan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat as Load

from IW_KRR import InstanceKRR
import Datasets
import Metrics

for ii in range(16):
    
    Xs,Ys,Xa,Ya,Xt,Yt = Datasets.Load_BM(BM_id = ii+1, aux_ratio = 0.05)
    #Xs,Ys,Xa,Ya,Xt,Yt,changed_id = Datasets.gen_noisy_regress_data(500,0.3)
    #read_file = Load("D7_50_Subset2_DITL.mat")
    #Xs = np.vstack((read_file['X21'],read_file['X22']))
    #Ys = np.vstack((read_file['y21'],read_file['y22']))
    #Xa = read_file['X11']
    #Ya = read_file['y11']
    #Xt = read_file['X12']
    #Yt = read_file['y12']
    
    
    kRR = InstanceKRR(lmbd = 1,gamma = 0.01)
    kRR.fit(Xs,Ys)
    kRR.Solve_alpha(Xa,Ya,eta = 0.08,lmbd2 = 0.5)
    alphas = kRR.alphas
    
    
    
    
    Yprd = kRR.predict(Xt)
    Ybsl = kRR.kRR_predict(Xt)
    Err = Metrics.NRMSE(Yt,Yprd)[0]
    ErrBsl = Metrics.NRMSE(Yt,Ybsl)[0]
    output_str = "Bio_Marker_"+str(ii+1)+":  || Baseline kRR error: "
    output_str += str(ErrBsl)[:5]
    output_str += '  ||  Instance-weighted kRR error: '
    output_str += str(Err)[:5]
    print(output_str)