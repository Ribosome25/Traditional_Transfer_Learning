# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:46:40 2019

@author: Ruibzhan
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import Datasets
import CORAL

X1, X2 = Datasets.gen_two_Gaussian_dist()
Y1 = X1[:,0]*0.6 + 0.2*np.power(X1[:,1],2) + 0.2
Y2 = X2[:,0]*0.6 + 0.2*np.power(X2[:,1],2) + 0.2
Y1 = Y1 + np.random.rand(Y1.shape[0])
Y2 = Y2 + np.random.rand(Y2.shape[0])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.scatter(X1[:,0],X1[:,1],label = 's')
ax1.scatter(X2[:,0],X2[:,1],label = 't')
ax1.legend()

Xs, Xt = CORAL.Coral(X1,X2)


ax2 = fig.add_subplot(122)
ax2.scatter(Xs[:,0],Xs[:,1],label = 's')
ax2.scatter(Xt[:,0],Xt[:,1],label = 't')
ax2.legend()

ax1.title.set_text('Two X normal distributions  =>')
ax2.title.set_text('After the CORAL matching')