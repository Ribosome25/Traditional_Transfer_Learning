# -*- coding: utf-8 -*-
"""
Evaluate Methods

Created on Sun Apr 28 16:20:38 2019

@author: Ruibzhan
"""
import numpy as np
import pandas as pd
import scipy.io
import sklearn.datasets
from sklearn.preprocessing import scale
from random import sample
from sklearn.model_selection import train_test_split as Split

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

def load_pics():
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']

    src, tar = 'data/' + domains[1], 'data/' + domains[2]
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    Xs, Ys, Xtg, Ytg = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    Xa,Xt,Ya,Yt = Split(Xtg,Ytg,test_size = 0.9,random_state = 2)
    return Xs,Ys.flatten(),Xa,Ya.flatten(),Xt,Yt.flatten()

def gen_noisy_regress_data():
    FtVecs, Values = sklearn.datasets.make_regression(n_samples=1000, n_features=20, n_informative=10, 
                                                 n_targets=1, bias=5.0, effective_rank=None, tail_strength=0.5, 
                                                 noise=10.0, shuffle=True, coef=False, random_state=2)
    Xs,Xt,Ys,Yt = Split(FtVecs,Values,test_size=0.2, random_state=2)
    np.random.seed(3)
    idChanged = np.random.randint(0,len(Ys),size = 100)
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

if __name__ == '__main__':
#    Xs,Ys,Xt,Yt = load_forrest()
    print(1)
    
