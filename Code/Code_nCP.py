#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:09:37 2019

@author: tdh
"""
def states_nCP(k,n):
    output_state_list=[]
    if n==1:
        output_state_list=[[n]]
    elif n==2:
        for j in range(k+1):
            output_state_list.append([j, k-j])
        return(output_state_list)
    else:
        for i in range (k+1):
            other_states=states_nCP(k-i,n-1)
            for state in other_states:
                state.append(i)
                output_state_list.append(state)
        return(output_state_list)

