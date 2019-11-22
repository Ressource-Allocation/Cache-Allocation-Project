#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:09:37 2019

@author: tdh
"""
def states_nCP(n,cache_capacity,states):
    if n==1:
        for i in range(cache_capacity+1):
            states[i]=[i]
        return(states)
    else:
       for i in range (cache_capacity + 1):
           states[i]=
           for i in range(cache_capacity - m+1):
               
               L.append([m, states_2CP(cache_capacity - m)[i][0], states_2CP(cache_capacity - m)[i][1]])

def create_Q_matrix(n,cache_capacity):
    Q_matrix= dict()
    states=states_nCP(n,cache_capcity)
    for state in states:
        for action in states:
            Q_matrix[state,action]=0
    

