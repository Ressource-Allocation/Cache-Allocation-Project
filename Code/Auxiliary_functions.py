#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:40:14 2019

@author: tdh
"""

#Auxiliary function list
import numpy as np

def find_max_array(array):
    """
    Returns the maximum value in an array and the list of the indexes of this value.

    Parameters
    ------------
    array : array
    
    Returns
    ----------
    [maxi, max_indexes]
    maxi: float
    maximum value in array
    max_indexes: list
    list of indexes of maximum value in array
    
    """

    
    max_indexes=[]
    maxi= array.max()
    for i in range(np.size(array)):
        if array[i]==maxi:
            max_indexes.append(i)
    return(maxi, max_indexes)


def find_max_list(vec):
    """
    Returns the maximum value in a vector and the list of the indexes of this value.

    Parameters
    ------------
    vec : List
    
    Returns
    ----------
    [maxi, argmax]
    maxi: float
    maximum value in list
    argmax: list
    list of indexes of maximum value in list
    
    """
    maxi=vec[0]
    argmax=[0]
    for i in range(len(vec)):  
        if vec[i]>maxi:
            maxi=vec[i]
            argmax=[i]
        elif vec[i]== max:
            argmax.append(i)
    
    return(maxi, argmax)
    
def list_sum(L):
    S=0
    for i in L:
        S+=i
    return(S)


def find_best_action(rewards, state_index):
    """
    Returns the list of actions to take (with best estimated total reward) given a certain state.

    Parameters
    ------------
    rewards : matrix
        Q-matrix (reward matrix)
    state_index : int
        The index of the state
    
    Returns
    ----------
    [best_reward, best_action_list]: 
        best_reward: float
            The estimated total reward one gets with the action selected
        best_action_list: list
            The index list of action to take
    """
    (best_reward, best_actions_list)=find_max_array(np.array(rewards)[:,state_index]) #if reward is already an array np.array has to be erased
    return(best_reward, best_actions_list)



