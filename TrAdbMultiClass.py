# -*- coding: utf-8 -*-
"""
Adaboost
Created on Mon Apr 29 10:57:31 2019

@author: Ruibzhan
"""
import numpy as np
import pandas as pd
import Datasets
import Metrics
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LRC
from sklearn.tree import DecisionTreeClassifier as TreeC
from sklearn import tree
import progressbar

#============================================================
def MultiTrAdap(Xs,Ys,Xa,Ya,Xt,Yt,nIters = 200):
    p = progressbar.ProgressBar()
    Xsa = np.concatenate((Xs,Xa))
    Ysa = np.concatenate((Ys,Ya))
    Ns = Ys.shape[0]
    Na = Ya.shape[0]
    
    Epss = []
    TestAcc = []
    TestPrd = {}
    Weights = {}
    AdaAcc = []
    Beta = 1/(1+np.sqrt(2*np.log(Ns)/nIters))
    Wsa = np.ones(Ns+Na)/(Ns+Na)
    p.start(nIters)
    for ni in range(nIters):
        Weights[ni] = Wsa
        # update P, train and predict
        Psa = Wsa/np.sum(Wsa)
        clf = RFC(n_estimators=5,criterion = 'entropy',max_depth=2)
    #    clf = LRC(solver = 'liblinear',multi_class='ovr')
    #    clf = LinearSVC(multi_class='ovr')
    #    clf = TreeC(splitter='best',max_depth=3)
        if 0:
            clf.fit(Xa,Ya,sample_weight = Psa[-Na:])
            YsPrd = clf.predict(Xs)
            RorWs = 1*(YsPrd != Ys)
            clf.fit(Xs,Ys,sample_weight = Psa[:Ns])
            YaPrd = clf.predict(Xa)
            RorWa = 1*(YaPrd != Ya)
            RorW = np.concatenate((RorWs,RorWa))
        else:
            clf.fit(Xsa,Ysa,sample_weight = Psa)
            YsaPrd = clf.predict(Xsa)
            # calculate the accuracy on XYa
            RorW = 1*(YsaPrd != Ysa)
    #    print(ni,np.where(RorW==1)[0])
        Eps = np.sum((Wsa*RorW)[-Na:])/np.sum(Wsa[-Na:])
        Epss.append(1-Eps)
        # adjust Eps
        if Eps>=0.4:
            Eps = 0.4
        elif Eps <= 0:
            Eps = 0.01
            
        # Weight update
        if 1:
            Alpha = np.sqrt(Eps/(1-Eps))
            Coef = np.concatenate((Beta*np.ones(Ns),(1/Alpha)*np.ones(Na)))
            wUpdate = np.power(Coef,RorW)
        else:
            Alpha = np.cbrt((1-Eps)/(1+Eps))
            Ct = 2.5*(1-Eps)
            Coef = np.concatenate((Ct*Beta*np.ones(Ns),Alpha*np.ones(Na)))
            wUpdate = np.power(Coef,-25*RorW/nIters)
        Wsa = Wsa*wUpdate
        # result & summary
        Yprd = clf.predict(Xt)
        TestPrd[ni] = Yprd
        TestAcc.append(Metrics.Accuracy(Yt,Yprd))
        
        clf.fit(Xa,Ya,sample_weight = Psa[-Na:])
        AdaAcc.append(Metrics.Accuracy(Yt,clf.predict(Xt)))
        p.update(ni+1)
    #    print(np.mean(Target))
    p.finish()
    return Weights,Epss,TestPrd,TestAcc,AdaAcc

#============================================================
   
idChanged = []
#Xs,Ys,Xa,Ya,Xt,Yt,idChanged = Datasets.gen_noisy_classi_data()
#Xs,Ys,Xa,Ya,Xt,Yt = Datasets.load_heart()
Xs,Ys,Xa,Ya,Xt,Yt = Datasets.load_pics()

nIters = 50

clf0 = RFC(n_estimators=5,criterion = 'entropy',max_depth=2)
clf0.fit(Xa,Ya)
Acc0 = Metrics.Accuracy(Yt,clf0.predict(Xt))

SPweights, Acc_auxi, All_test_prd, Acc_test,Acc_AdaOnly = MultiTrAdap(Xs,Ys,Xa,Ya,Xt,Yt,nIters = nIters)

PrdDf = pd.DataFrame.from_dict(All_test_prd)    
HalfDf = PrdDf.iloc[:,round(nIters/2):]
BoostPrd = HalfDf.mode(axis = 1)
AccB = Metrics.Accuracy(Yt,BoostPrd[0])

import matplotlib.pyplot as plt
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(Acc_test,label = 'From S+A to Test')
ax1.plot(Acc_auxi,label = 'Weighted Aux set')
#ax1.plot(Acc_AdaOnly,label = 'Use boosted Aux only')
plt.legend(loc='best', shadow=True,fontsize = 'x-large')
plt.xlabel('n of iterations')
plt.ylabel('Accuracy')
plt.title("Boosting accuracy = %.4f"%AccB,fontsize = 'x-large')
plt.show()

WghtDf = pd.DataFrame.from_dict(SPweights)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for ni in range(30):
    ii = np.random.randint(0,len(Ys)+len(Ya))
    if len(idChanged)>0 and (ii in idChanged):
        ax2.semilogy(WghtDf.loc[ii],color = 'red',linewidth = 0.5)
    elif ii<len(Ys):
        ax2.semilogy(WghtDf.loc[ii],color = 'blue',linewidth = 0.5)
    else:
        ax2.semilogy(WghtDf.loc[ii],color = 'green',linewidth = 0.5)
plt.title('Sample weights change (Semi-log scale)')
#  '20 feature vectors are randomly picked to show the evolution of sample weights: red ones indicates the wrongs samples, blue ones indicate the unchanged samples')
plt.show()






