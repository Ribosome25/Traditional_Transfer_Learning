# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:12:41 2019

@author: Ruibzhan

Generate the parameter combinations from given parameter lists.
i.e. from n-dim to 1-dim. Finding out all the combinations

Example:
Paras = {'kern': ['lin','rbf'], 'sigma':np.logspace(-1,1,10), 'B': np.linspace(1,5,5), 'lmbd' : [0,1]}
Para_names = list(Paras.keys())
Para_table = pd.DataFrame(GridSearchTL.get_comb(Paras,Para_names),columns = Para_names)

"""
import numpy as np

#%%
#Para_names = list(Paras.keys())

def get_comb(Paras,Para_names):

    if len(Para_names) == 1:
        return np.vstack(Paras[Para_names[0]])
    
    this_para = Para_names[-1]
    temp_array = []
    for each_value in Paras[this_para]:
        get_array = get_comb(Paras,Para_names[:-1])
        length = len(get_array)
        attach_array = np.repeat(np.array(each_value).reshape(-1,1),length,axis = 0)
        return_array = np.hstack((get_array,attach_array))
        temp_array.append(return_array)
    return np.vstack(temp_array)
    

