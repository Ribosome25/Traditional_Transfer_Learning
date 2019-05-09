# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:54:19 2019

@author: Ruibzhan
"""
import numpy as np

def NRMSE(Y_Target, Y_Predict, multi_dimension = False):
    Y_Target = np.array(Y_Target); Y_Predict = np.array(Y_Predict);
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target),1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict),1)
    Y_Bar = np.mean(Y_Target)
    Nom = np.sum((Y_Predict - Y_Target)**2)
    Denom = np.sum((Y_Bar - Y_Target)**2)
    MSE = np.mean((Y_Predict - Y_Target)**2)
    NRMSE_Val = np.sqrt(Nom/Denom)
    return NRMSE_Val, MSE


def Accuracy(Y_Target,Y_Predict,error_rate = False,multi_dimension = False):
    Y_Target = np.asarray(Y_Target,order = 'C')
    Y_Predict = np.asarray(Y_Predict,order = 'C')
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target),1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict),1)
    if len(Y_Target)==len(Y_Predict):
        correct = np.sum(Y_Target==Y_Predict)
        if error_rate:
            Value = 1-(correct/len(Y_Predict))
        else:
            Value = correct/len(Y_Predict)
    else:
        raise ValueError ('Target & Predicted are not of same length.')
        return np.nan
    
    return Value

            
    
    
    
    