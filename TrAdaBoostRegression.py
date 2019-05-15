from scipy.io import loadmat as Load
import numpy as np
import pandas as pd
import Metrics
import progressbar

#--------Models-------------
from sklearn.tree import DecisionTreeRegressor as RTree
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import Ridge as LR
from sklearn.ensemble import RandomForestRegressor as RFR
#-------Functions-----------
def Load_BM(BM_id = 1,test_ratio = 0.01):
    file_name = '.\data\MB_'+str(BM_id)+'.mat'
    file_input = {}
    Load(file_name,mdict=file_input)
    Xs = file_input['X2']
    Ys = file_input['Y2']
    np.random.seed(15)#=-=-=-
    test_id = np.random.randint(0,high = file_input['X1'].shape[0], size = int(test_ratio*file_input['X1'].shape[0]) )
    sampling_mask = np.zeros(file_input['X1'].shape[0], dtype=bool)
    sampling_mask[test_id] = True
    Xa = file_input['X1'][sampling_mask,:]
    Ya = file_input['Y1'][sampling_mask,:]
    sampling_mask = np.invert(sampling_mask)
    Xt = file_input['X1'][sampling_mask,:]
    Yt = file_input['Y1'][sampling_mask,:]
    return Xs,Ys,Xa,Ya,Xt,Yt

Xs,Ys,Xa,Ya,Xt,Yt = Load_BM(3)
Xat = np.concatenate((Xa,Xt))
Yat = np.concatenate((Ya,Yt))
np.random.seed(15) #=-=-
BL_Mdl = RFR(n_estimators=200,min_samples_leaf = 5)
BL_Mdl.fit(Xs,Ys.ravel())
Yprd = BL_Mdl.predict(Xat)
print("\nFrom Source to A+T: \n",Metrics.NRMSE(Yat,Yprd)[0])
print('===')
BL_Mdl.fit(Xa,Ya.ravel())
Yprd = BL_Mdl.predict(Xt)
print("\nFrom A to T: \n",Metrics.NRMSE(Yt,Yprd)[0])
print('===')

def RegreTrAdap(Xs,Ys,Xa,Ya,Xt,Yt,nIters = 200,Temp_Para = 20):
    p = progressbar.ProgressBar()
    p.start(nIters)
    
    Xsa = np.concatenate((Xs,Xa))
    Ysa = np.concatenate((Ys,Ya)).flatten()
    Ns = Ys.shape[0]
    Na = Ya.shape[0]
    
    Epss = []
    TestAcc = []
    TestPrd = {}
    Weights = {}
    AdaAcc = []
        
    Beta = 1/(1+np.sqrt(2*np.log(Ns)/nIters))
    Wsa = np.ones(Ns+Na)/(Ns+Na)
    
    
    for ni in range(nIters):
        Weights[ni] = Wsa
        Psa = Wsa/np.sum(Wsa)
        #------------MDL selection-------------#
#        clf = LR(alpha = 0.1)
        clf = RTree(min_samples_leaf=5)
#        clf = RFR(n_estimators=10,min_samples_leaf = 5)
        #-----------Mdl fit+predict-----------#
        clf.fit(Xsa,Ysa.ravel(),sample_weight=Psa)
        YsaPrd = clf.predict(Xsa)
        #-----------Error calculation----------#
#        Errors = np.abs(Ysa-YsaPrd)/np.std(Ysa) # used the std instead of max
        Errors = np.abs(Ysa-YsaPrd)/np.max(abs(Ysa-YsaPrd))
        Eps = sum(Errors[-Na:]*Wsa[-Na:])/sum(Wsa[-Na:])
        Epss.append(Eps)
        
        if Eps>0.5:
            Eps = 0.40
        elif Eps==0:
            Eps = 0.01
        #--------Weight update---------------#
        if 1:
            Alpha = np.sqrt(Eps/(1-Eps))
#            Alpha = Eps/(1-Eps)
            Coef = np.concatenate((Beta*np.ones(Ns),(1/Alpha)*np.ones(Na)))
            wUpdate = np.power(Coef,50*Errors/nIters)
        else:
            Alpha = np.sqrt((1-Eps)/(1+Eps))
            Ct = 1.5/Alpha
            Coef = np.concatenate((Ct*Beta*np.ones(Ns),Alpha*np.ones(Na)))
            wUpdate = np.power(Coef,-30*Errors/nIters)
            
        Wsa = Wsa*wUpdate
        Yprd = clf.predict(Xt)
        TestPrd[ni] = Yprd
        TestAcc.append(Metrics.NRMSE(Yt,Yprd)[0])
    #    print(Metrics.NRMSE(Ya,Yp[-Na:])[0])
        clf.fit(Xa,Ya.ravel(),sample_weight = Psa[-Na:])
        AdaAcc.append(Metrics.NRMSE(Yt,clf.predict(Xt))[0])
        
        p.update(ni+1)
    p.finish()

    return Weights,Epss,TestPrd,TestAcc,AdaAcc


nIters = 100
SPweights, Acc_auxi, All_test_prd, Acc_test,Acc_AdaOnly = RegreTrAdap(Xs,Ys,Xa,Ya,Xt,Yt,nIters = nIters)

PrdDf = pd.DataFrame.from_dict(All_test_prd)    
HalfDf = PrdDf.iloc[:,round(nIters/2):]
HalfWeights = 1-pd.DataFrame(Acc_auxi[round(nIters/2):])
HalfWeights = HalfWeights/HalfWeights.sum()
BoostPrd = np.dot(HalfDf,HalfWeights)
AccB = Metrics.NRMSE(Yt,BoostPrd)[0]
print('\n==>>',Z,AccB)

#===================Plot=====================
import matplotlib.pyplot as plt
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(Acc_test,label = 'Test set error')
ax1.plot(Acc_auxi,label = 'Assistant Epsi')
ax1.plot(Acc_AdaOnly,label = 'Adaboost only')
plt.legend(loc='best', shadow=True,fontsize = 'x-large')
plt.xlabel('n of iterations')
plt.ylabel('NRMSE')
plt.title("Boosting Error = %.4f"%AccB)
plt.show()
idChanged = []
WghtDf = pd.DataFrame.from_dict(SPweights)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for ni in range(100):
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


