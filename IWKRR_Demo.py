# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:44:01 2019

@author: Ruibzhan
"""
import numpy as np
import matplotlib.pyplot as plt

from IW_KRR import InstanceKRR
    
#============================
rng = np.random.RandomState(0)
# Generate sample data
X = 15 * rng.rand(100, 1)
y = np.sin(X).ravel()
y += 2 * (0.5 - rng.rand(X.shape[0]))

rng = np.random.RandomState(1)
Xa = 15 * rng.rand(25, 1)
ya = np.sin(Xa-1).ravel()
ya += (0.5 - rng.rand(Xa.shape[0])) 

rng = np.random.RandomState(2)
Xt = 15 * rng.rand(50, 1)
yt = np.sin(Xt-1).ravel()

X_plot = np.linspace(0, 20, 10000)[:, None]

    


    
krr = InstanceKRR(lmbd = 0.5)
krr.fit(X,y)
yhat = krr.kRR_predict(X_plot)

krr.Solve_alpha(Xa,ya)
y_prd = krr.predict(X_plot)







# Plot results
plt.figure(figsize=(10, 5))
lw = 2
plt.scatter(X, np.array(y), c='k', label='data w\ noise')
plt.plot(X_plot, np.array(yhat), c='g', lw = 5, label='KRR')

plt.scatter(Xa, ya, c = 'r', label = "Auxilary data")
plt.plot(X_plot, y_prd, c = 'r', lw = 5, label = "Instance weighted KRR")
plt.plot(X_plot, np.sin(X_plot), color='navy', lw=lw, label='True value')
plt.legend()
plt.show()