# -*- coding: utf-8 -*-
"""
Evaluate Methods

Created on Sun Apr 28 16:20:38 2019

@author: Ruibzhan
"""
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import loadmat as Load

import sklearn.datasets
from sklearn.preprocessing import scale
from random import sample
from sklearn.model_selection import train_test_split as Split

#-------------Regression Sets--------------------
##------------- Lung Cancer ------------
def Load_Lung_CCLE(BM_id = 1,aux_ratio = 0.0, stdize = False):
    # Processed Tumor/Cellines
    file_name = '.\data\\NSCLC_Source_CCLE\\Data_BM_'+str(BM_id)+'_f150_CCLE_NSCLC.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    Xtgt = file_input['X1']
    Ytgt = file_input['Y1']
    if stdize:
        n_source = Xs.shape[0]
        Xconc = np.vstack((Xs,Xtgt))
        Xconc = scale(Xconc)
        Xs = Xconc[:n_source,:]
        Xtgt = Xconc[n_source:,:]
        
    np.random.seed(15)#    <<=== Change
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = Xtgt[sampling_mask,:]
    Ya = Ytgt[sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = Xtgt[sampling_mask,:]
    Yt = Ytgt[sampling_mask,:]
    
    return Xs,Ys,Xa,Ya,Xt,Yt

def Load_Lung_CCLE_Orig(BM_id = 1,aux_ratio = 0.0, stdize = False):
    # Processed Tumor/Cellines
    file_name = '.\data\\NSCLC_Source_CCLE - Orig\\Data_BM_'+str(BM_id)+'_f150_CCLE_NSCLC.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    Xtgt = file_input['X1']
    Ytgt = file_input['Y1']
    if stdize:
        n_source = Xs.shape[0]
        Xconc = np.vstack((Xs,Xtgt))
        Xconc = scale(Xconc)
        Xs = Xconc[:n_source,:]
        Xtgt = Xconc[n_source:,:]
        
    np.random.seed(15)#    <<=== Change
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = Xtgt[sampling_mask,:]
    Ya = Ytgt[sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = Xtgt[sampling_mask,:]
    Yt = Ytgt[sampling_mask,:]
    
    return Xs,Ys,Xa,Ya,Xt,Yt

def Load_Lung_Both(BM_id = 1,aux_ratio = 0.0,stdize = False):
    # Processed Tumor/Cellines
    file_name = '.\data\\NSCLC_Source_CCLE_GDSC\\Data_BM_'+str(BM_id)+'_f150_CCLE_GDSC_NSCLC.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    Xtgt = file_input['X1']
    Ytgt = file_input['Y1']
    if stdize:
        n_source = Xs.shape[0]
        Xconc = np.vstack((Xs,Xtgt))
        Xconc = scale(Xconc)
        Xs = Xconc[:n_source,:]
        Xtgt = Xconc[n_source:,:]
        
    np.random.seed(15)#    <<=== Change
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = Xtgt[sampling_mask,:]
    Ya = Ytgt[sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = Xtgt[sampling_mask,:]
    Yt = Ytgt[sampling_mask,:]
    
    return Xs,Ys,Xa,Ya,Xt,Yt

def Load_Lung_Both_Orig(BM_id = 1,aux_ratio = 0.0,stdize = False):
    # Processed Tumor/Cellines
    file_name = '.\data\\NSCLC_Source_CCLE_GDSC - Orig\\Data_BM_'+str(BM_id)+'_f150_CCLE_GDSC_NSCLC.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    Xtgt = file_input['X1']
    Ytgt = file_input['Y1']
    if stdize:
        n_source = Xs.shape[0]
        Xconc = np.vstack((Xs,Xtgt))
        Xconc = scale(Xconc)
        Xs = Xconc[:n_source,:]
        Xtgt = Xconc[n_source:,:]
        
    np.random.seed(15)#    <<=== Change
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = Xtgt[sampling_mask,:]
    Ya = Ytgt[sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = Xtgt[sampling_mask,:]
    Yt = Ytgt[sampling_mask,:]
    
    return Xs,Ys,Xa,Ya,Xt,Yt

def Load_Lung_GDSC(BM_id = 1,aux_ratio = 0.0,stdize = False):
    # Processed Tumor/Cellines
    file_name = '.\data\\NSCLC_Source_GDSC\\Data_BM_'+str(BM_id)+'_f150_GDSC_NSCLC.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    Xtgt = file_input['X1']
    Ytgt = file_input['Y1']
    if stdize:
        n_source = Xs.shape[0]
        Xconc = np.vstack((Xs,Xtgt))
        Xconc = scale(Xconc)
        Xs = Xconc[:n_source,:]
        Xtgt = Xconc[n_source:,:]
        
    np.random.seed(15)#    <<=== Change
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = Xtgt[sampling_mask,:]
    Ya = Ytgt[sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = Xtgt[sampling_mask,:]
    Yt = Ytgt[sampling_mask,:]
    
    return Xs,Ys,Xa,Ya,Xt,Yt

##------------ Breast Cancer------------
def Load_BM(BM_id = 1,aux_ratio = 0.9):
    # Processed Tumor/Cellines
    file_name = '.\data\stdX\MB_'+str(BM_id)+'.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    np.random.seed(15)#    <<===
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = file_input['X1'][sampling_mask,:]
    Ya = file_input['Y1'][sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = file_input['X1'][sampling_mask,:]
    Yt = file_input['Y1'][sampling_mask,:]
    return Xs,Ys,Xa,Ya,Xt,Yt

def Load_BM_raw(BM_id = 1,aux_ratio = 0.015):
    # Processed Tumor/Cellines
    file_name = '.\\data\\rawX\\Data_BM_raw_'+str(BM_id)+'.mat'
    # raw later
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    np.random.seed(15)#    <<===
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = file_input['X1'][sampling_mask,:]
    Ya = file_input['Y1'][sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = file_input['X1'][sampling_mask,:]
    Yt = file_input['Y1'][sampling_mask,:]
    return Xs,Ys,Xa,Ya,Xt,Yt

def Load_BM_scaled_raw(BM_id = 1,aux_ratio = 0.015):
    # Processed Tumor/Cellines
    file_name = '.\\data\\rawX\\Data_BM_raw_'+str(BM_id)+'.mat'
    # S
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    # Sample
    np.random.seed(15)#    <<===
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(aux_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = file_input['X1'][sampling_mask,:]
    Ya = file_input['Y1'][sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = file_input['X1'][sampling_mask,:]
    Yt = file_input['Y1'][sampling_mask,:]
    
    _meanS = np.mean(Xs,axis = 0).reshape(1,-1)
    _stdS = np.std(Xs,axis = 0).reshape(1,-1)
    Xs -= _meanS
    Xa -= _meanS
    Xt -= _meanS
    Xs /= _stdS
    Xa /= _stdS
    Xt /= _stdS
        
    return Xs,Ys,Xa,Ya,Xt,Yt

def Load_BM_reverse(BM_id = 1,aux_ratio = 0.015):
    # Processed Tumor/Cellines
    file_name = '.\\data\\rawX\\Data_BM_'+str(BM_id)+'.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X1']
    Ys = file_input['Y1']
    # Sample
    np.random.seed(15)#    <<===
    test_id = np.random.randint(0,high = file_input['X2'].shape[0], size = int(aux_ratio*file_input['X2'].shape[0]) )
    sampling_mask = np.zeros(file_input['X2'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = file_input['X2'][sampling_mask,:]
    Ya = file_input['Y2'][sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = file_input['X2'][sampling_mask,:]
    Yt = file_input['Y2'][sampling_mask,:]
    
    return Xs,Ys,Xa,Ya,Xt,Yt
    
def Load_BM_raw_reverse(BM_id = 1,aux_ratio = 0.015):
    # Processed Tumor/Cellines
    file_name = '.\\data\\rawX\\Data_BM_raw_'+str(BM_id)+'.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X1']
    Ys = file_input['Y1']
    # Sample
    np.random.seed(15)#    <<===
    test_id = np.random.randint(0,high = file_input['X2'].shape[0], size = int(aux_ratio*file_input['X2'].shape[0]) )
    sampling_mask = np.zeros(file_input['X2'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = file_input['X2'][sampling_mask,:]
    Ya = file_input['Y2'][sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = file_input['X2'][sampling_mask,:]
    Yt = file_input['Y2'][sampling_mask,:]
    
    return Xs,Ys,Xa,Ya,Xt,Yt

# ------------- Toy data ---------------
def gen_two_Gaussian_dist():
    np.random.seed(15)
    X1 = np.random.multivariate_normal([1,1],[[1,0.5],[0.5,1]],200)
    X2 = np.random.multivariate_normal([1,1],[[5,-3],[-3,5]],200)
    return X1,X2


def load_drug():
    ReadData = scipy.io.loadmat('data\DrugData.mat')
    Xs,Xt,Ys,Yt = ReadData['Xc'],ReadData['Xm'],ReadData['Yc'],ReadData['Ym']
    Xa, Xt, Ya, Yt = Split(Xt, Yt, test_size=0.9, random_state=42)
    return Xs,Ys,Xa,Ya,Xt,Yt

def load_wine():
    White = pd.read_csv("data\winequality-white.csv",sep = ';')
    Red = pd.read_csv("data\winequality-red.csv",sep = ';')
    Xs = White.iloc[:,:-1]
    Ys = White.iloc[:,-1]
    Ta, Tt = Split(Red,test_size = 0.98,random_state = 10)
    Xa = Ta.iloc[:,:-1]
    Ya = Ta.iloc[:,-1]
    Xt = Tt.iloc[:,:-1]
    Yt = Tt.iloc[:,-1]
    return Xs,Ys,Xa,Ya,Xt,Yt

def gen_noisy_regress_data(n_samples = 200, AT_S_ratio = 0.4):
    FtVecs, Values = sklearn.datasets.make_regression(n_samples=n_samples, n_features=20, n_informative=10, 
                                                 n_targets=1, bias=5.0, effective_rank=None, tail_strength=0.5, 
                                                 noise=10.0, shuffle=True, coef=False, random_state=2)
    Xs,Xt,Ys,Yt = Split(FtVecs,Values,test_size=AT_S_ratio, random_state=2)
    np.random.seed(3)
    idChanged = np.random.randint(0,len(Ys),size = 20)
    Ys[idChanged] = np.random.permutation(Ys[idChanged])
    Xa,Xt,Ya,Yt = Split(Xt,Yt,test_size = 0.8,random_state = 2)
    return(Xs,Ys,Xa,Ya,Xt,Yt,idChanged)
    
def gen_Friedman1_data():
    d = 1
    tempDictX = []
    tempDictY = []
    for ii in range(5):
        aa = np.random.normal(loc = 1, scale = 0.1*d, size = 4)
        bb = np.random.normal(loc = 1, scale = 0.1*d, size = 5)
        cc = np.random.normal(loc = 0, scale = 0.05*d, size = 5)
        Xs = np.random.normal(loc = 0, scale = 1, size = (200,10))
        Ys = aa[0]*10*np.sin(np.pi*(bb[0]*Xs[:,0]+cc[0])*(bb[1]*Xs[:,1]+cc[1])) +\
        aa[1]*20*(bb[2]*Xs[:,2]+cc[2]-0.5)**2 + aa[2]*10*(bb[3]*Xs[:,3]+cc[3]) + \
        aa[3]*5*(bb[4]*Xs[:,4]+cc[4])
        Ys = Ys + np.random.normal(size = Ys.shape)
        tempDictX.append(Xs)
        tempDictY.append(Ys)
    Xs = np.concatenate(tempDictX)
    Ys = np.concatenate(tempDictY)
    Xtg = np.random.normal(loc = 0, scale = 1, size = (1000,10))
    Ytg = 10*np.sin(np.pi*Xtg[:,0]*Xtg[:,1]) + 20*(Xtg[:,2]-0.5)**2 + 10*Xtg[:,3] + 5*Xtg[:,4]
    Ytg = Ytg + np.random.normal(size = Ytg.shape)
    Xa,Xt,Ya,Yt = Split(Xtg,Ytg,test_size = 0.9,random_state = 2)
    
    return Xs,Ys,Xa,Ya,Xt,Yt

#------------------Multi Class Classifications--------------------

def load_forrest():
    FrstData1 = pd.read_excel('data\Area3.xlsx')
    FrstData2 = pd.read_excel('data\Area4.xlsx')

    FrstData_s1 = FrstData1.sample(n=None, frac=0.25, replace=False, weights=None, random_state=2, axis=None)
    FrstData_s2 = FrstData2.sample(n=None, frac=0.05, replace=False, weights=None, random_state=2, axis=None)
    FrstData_s = pd.concat((FrstData_s1,FrstData_s2))
    FrstData_t = FrstData2.sample(n=None, frac=0.03, replace=False, weights=None, random_state=2, axis=None)
    
    Xs = scale(FrstData_s.iloc[:,:10])
    Ys = np.asarray(FrstData_s.iloc[:,-1])
    
    TestAmount = 1060
    Xa = scale(FrstData_t.iloc[:-TestAmount,:10])
    Ya = np.asarray(FrstData_t.iloc[:-TestAmount,-1])
    Xt = scale(FrstData_t.iloc[-TestAmount:,:10])
    Yt = np.asarray(FrstData_t.iloc[-TestAmount:,-1])
    return Xs,Ys,Xa,Ya,Xt,Yt


def load_heart():
    Cols = [ii for ii in range(10)]
    Cols.append(-1)
    Data1 = pd.read_csv('data\Heartdata2.csv',na_values = '?').iloc[:,Cols]
    Data2 = pd.read_csv('data\Heartdata1.csv',na_values = '?').iloc[:,Cols]
    Data2 = Data2.dropna()
    Data_a, Data_t = Split(Data2,test_size = 0.95,random_state = 5)
    Xs = Data1.iloc[:,:-1]
    Ys = Data1.iloc[:,-1]
    Xa = Data_a.iloc[:,:-1]
    Ya = Data_a.iloc[:,-1]
    Xt = Data_t.iloc[:,:-1]
    Yt = Data_t.iloc[:,-1]
    return Xs,Ys,Xa,Ya,Xt,Yt
    
    
def gen_noisy_classi_data():
    FtVecs ,Labels = sklearn.datasets.make_classification(n_samples=500, n_features=5, n_informative=3, 
                                         n_redundant=2, n_repeated=0, n_classes=3, 
                                         n_clusters_per_class=2, weights=None, flip_y=0.01, 
                                         class_sep=1.0, hypercube=True, shift=0.0, scale=1.1, 
                                         shuffle=True, random_state=2)
    Xs,Xt,Ys,Yt = Split(FtVecs,Labels,test_size=0.2, random_state=2)
    
    Changed = Ys == 1
    np.random.seed(3)
    Ys[Ys==1] = np.random.randint(0,high = 2,size = (len(Ys[Ys==1])))
    Changed = Changed * [Ys == 0]
    idChanged = np.where(Changed)[1]
    print('Changed Idx:',idChanged)

    Xa,Xt,Ya,Yt = Split(Xt,Yt,test_size = 0.9,random_state = 2)
    
    return(Xs,Ys,Xa,Ya,Xt,Yt,idChanged)
    

def load_pics():
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']

    src, tar = 'data/' + domains[1], 'data/' + domains[2]
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    Xs, Ys, Xtg, Ytg = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    Xa,Xt,Ya,Yt = Split(Xtg,Ytg,test_size = 0.9,random_state = 2)
    return Xs,Ys.flatten(),Xa,Ya.flatten(),Xt,Yt.flatten()


if __name__ == '__main__':
#    Xs,Ys,Xt,Yt = load_forrest()
    print('Data Sets ')
    
