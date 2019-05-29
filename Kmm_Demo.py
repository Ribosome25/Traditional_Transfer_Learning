# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:39:25 2019

@author: Ruibzhan
"""
import numpy as np 
import matplotlib.pyplot as plt

import KMM

from sklearn.linear_model import LinearRegression as LR

#=======Data generation================
rng = np.random.RandomState(1)
X1 = rng.normal(loc = 0.5, scale = 0.5, size = 100)
Y1 = -X1 + np.power(X1,3)
Y1 += rng.normal(0,0.3,size = Y1.shape)

rng = np.random.RandomState(2)
Xp = rng.normal(loc = 0, scale = 0.3, size = (50,1))
Yp = -Xp + np.power(Xp,3)
Yp += rng.normal(0,0.3,size = Yp.shape)

#======================
kmm = KMM.kmm(Xp,X1)
#kmm.fit_kernel_mean_matching(kern='lin',B = 5,lmbd = 1)
kmm.fit_featurewise_kernel_mean_matching(kern='lin',B = 5,lmbd = 1)
coef = kmm.coef

#====================
lr_1 = LR()
lr_2 = LR()
lr_3 = LR()
lr_1.fit(X1.reshape(-1,1),Y1,sample_weight=None)
lr_2.fit(Xp.reshape(-1,1),Yp)
lr_3.fit(X1.reshape(-1,1),Y1,sample_weight=coef.ravel())

#======= Plot ========
_X_plot = np.linspace(-1,2,num = 20)
_Y_true = -_X_plot + np.power(_X_plot,3)

plt.close()
plt.figure()
plt.plot(_X_plot,_Y_true,lw = 1)
plt.scatter(X1, Y1, color='b', marker='+')
plt.scatter(Xp, Yp, color='r', marker='+')

Y_unweighted = lr_1.predict(_X_plot.reshape(-1,1))
Y_weighted = lr_3.predict(_X_plot.reshape(-1,1))
Y_target = lr_2.predict(_X_plot.reshape(-1,1))
plt.scatter(X1, Y1, color='green', s=coef*50, alpha=0.2)
plt.plot(_X_plot,Y_unweighted,lw = 1,c = 'b',ls = '--')
plt.plot(_X_plot,Y_weighted,lw = 1, c = 'g', ls = '--')
plt.plot(_X_plot,Y_target,lw = 1, c = 'r', ls = '--')