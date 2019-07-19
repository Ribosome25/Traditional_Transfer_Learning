# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:12:41 2019
Generate the parameter combinations from given parameter lists.
i.e. from n-dim to 1-dim. Finding out all the combinations
@author: Ruibzhan
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
    

