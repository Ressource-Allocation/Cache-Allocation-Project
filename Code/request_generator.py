#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:05:53 2019

@author: tdh
"""

import random as rd

#We create the list of probabilities to choose one of the k content providers"
def list_of_probas(k):
    list_proba=[]
    S=0
    for i in range(1, k+1):
        r=rd.random()
        S=S+r
        list_proba.append(r)
    for j in range(0, k):
        list_proba[j]=list_proba[j]/S
    return list_proba
        

#LUniform distribution probablility list
def uniform_distrib_list(k):
    list_proba=[]
    for i in range(k):
        list_proba.append(1/k)
    return list_proba


#We create the alpha list for the k CP
def alphas_list(k):
    alpha_list=[]
    for i in range(k):
        alpha_list.append(0.8)
    return alpha_list


#List of number of videos by CP
def video_nb_list(k):
    video_list=[]
    for i in range(k):
        video_list.append(1000)
    return video_list


#Nomalized probabilities list
def proba_seed_list(k):
    seedlist=[]
    S=0
    for i in range(k):
        seedlist.append(rd.random())
        S += seedlist[i]
    for p in seedlist:
        p /= S  #normalisation pour S=1
    return seedlist


#test creation alpha list with seed(1) with 3 alpha values (0.8, 1.0, 1.2)
def alpha_seed1_list(k):
    rd.seed(1)
    l=[]
    for i in range(k):
        r=rd.random()
        if r<1/3:
            l.append(0.8)
        elif r<2/3:
            l.append(1.0)
        else:
            l.append(1.2)
    return l               #returns always the same list


#test creation alpha list with seed(2) with 3 alpha values (0.8, 1.0, 1.2)
def alpha_seed2_list(k):
    rd.seed(2)
    l=[]
    for i in range(k):
        r=rd.random()
        if r<1/3:
            l.append(0.8)
        elif r<2/3:
            l.append(1.0)
        else:
            l.append(1.2)
    return l               #returns always the same list
        

#test creation alpha list with seed(3) with 3 alpha values (0.8, 1.0, 1.2)
def alpha_seed3_list(k):
    rd.seed(3)
    l=[]
    for i in range(k):
        r=rd.random()
        if r<0.333333:
            l.append(0.8)
        elif r<0.666666:
            l.append(1.0)
        else:
            l.append(1.2)
    return l               #returns always the same list


#Create lists of fix numbers (100, 1000, 10000) of videos for each k CP
def list_100_videos(k):
    l=[]
    for i in range(k):
        l.append(100)
    return l
        


def list_1000_videos(k):
    l=[]
    for i in range(k):
        l.append(1000)
    return l



def list_10000_videos(k):
    l=[]
    for i in range(k):
        l.append(10000)
    return l