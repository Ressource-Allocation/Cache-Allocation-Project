#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:47:50 2019

@author: tdh
"""

import random as rd
import numpy as np
import Auxiliary_functions as af #To simplify code
from copy import deepcopy #allow to make independant copies of lists (not just aliases)
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def init():
    """
    Initialize global variables 
    """    
    global list_alpha #list of alphas used in zipf functions for each SP
    global SP_proba #list of popularity of each SP (probability of requesting a CP)
    global video_nb_list #Number of videos provided by each SP
    global conss_zipf #list of computed norms for each SP whith parameters in list_alpha and for number of videos in video_nb_list above. Used to optimize computation time
    global nSP #Number of SPs
    global cache_capacity #Cache capacity
    global nb_videos #nb of videos for each SP
    global nb_bins #nb of bins for each SP where videos are stored
    global gamma #equation parameter to compute the new Q in SARSA
    global epsilon #epsilon-greedy politic
    global alpha_de_sarsa #equation parameter to compute the new Q in SARSA
    #global DEBUG
    rd.seed(5)
    list_alpha=[0.8, 1.0, 1.2]
    SP_proba=[0.7, 0.25, 0.05] #test try and find convergence
    nSP=3
    cache_capacity=10**3
    nb_bins=10**3
    nb_videos=10**6
    video_nb_list=[nb_videos for i in range(nSP)]
    gamma = 0.4
    epsilon = 0.2
    alpha_de_sarsa = 0.9
    conss_zipf=[af.zipf_norm(list_alpha[0],nb_videos), af.zipf_norm(list_alpha[1],nb_videos), af.zipf_norm(list_alpha[2],nb_videos)]

    #DEBUG=False

init() 


def zipf_distribution(alpha, nb_videos, norm):
    """
    Creates a list of the discrete probabilities for a catalog of nb_videos videos following a Zipf law with parameter Alpha.
    Parameters
    ------------
    alpha: float
        Parameter of Zipf law
    nb_videos: int
        Number of videos in the catalog
    norm: float
        Computed norm to normalise probabilities (computed with af.zipf_norm)
    
    Returns
    ------------
    probabilities_pi: list of float
        List of the probabilities of requesting the video represented by it index on the list 
    """
    probabilites_pi = np.zeros(nb_videos)
    for i in range(1, nb_videos+1):
        pi = (1.0/i**alpha) * (1.0/norm)
        probabilites_pi[i-1] = pi
    return probabilites_pi

def catalog():
    """
    Creates the catalog of videos and the catalog of bins for all SPs. Bins are groups of videos used to optimize time computation of the creation of request.
    
    Parameters
    ------------
    Uses global variables:
   
    nSP: int 
        number of SPs
    list_alpha: list of float
        list of alphas used in zipf functions for each SP
    video_nb_list: list of int
        Number of videos provided by each SP
    conss_zipf: list of float
        list of computed norms for each SP
    Returns
    ------------
    videos: list of list of float
        list of lists of probabilities of requesting a video for each SP
    bins:
        list of lists of probabilities of requesring a video in a bin for each SP
    """
    videos=np.zeros((nSP,nb_videos))
    bins=np.zeros((nSP,10**3))
    for SP in range(nSP):
        videos[SP] = zipf_distribution(list_alpha[SP], video_nb_list[SP], conss_zipf[SP])
        for bin_number in range(10**3):
            for video_number in range(10**3):
                bins[SP][bin_number]+=videos[SP][bin_number*10**3+video_number]
    return(videos,bins)


def request_creation(video_probability,bin_probability):
    """
    Returns indexes of the selected content provider and the selected video in a simulated request.

    Parameters
    ------------ 
    video_probability: list of list of float
        list of lists of probabilities of requesting a video for each SP
    bin_probability:
        list of lists of probabilities of requesting a video in a bin for each SP
    
    Uses global variables:
    
    SP_proba: list of float
        list of popularity of each SP (probability of requesting a CP)
    
    Returns
    ----------
    [selected_SP, selected_video]
    selected_SP: int
        Index of the selected content provider
    selected_video: int
        Index of the selected video
    """
    S=SP_proba[0]
    selected_SP=0
    SP_choice = rd.random()
    while(SP_choice>S):
        selected_SP+=1
        S+=SP_proba[selected_SP]
    video_choice = rd.random()
    S1=bin_probability[selected_SP][0]
        
    selected_bin=0
    S2=0
    while(video_choice>S1):
        selected_bin+=1
        S2=S1
        S1+=bin_probability[selected_SP][selected_bin]
    selected_video=(selected_bin*10**3)-1
    while(video_choice>S2):
        selected_video+=1
        S2+=video_probability[selected_SP][selected_video]
    #if DEBUG:
     #   print('selected_SP is', selected_SP)
      #  print('selected_video is', selected_video)"""
    return [selected_SP, selected_video]


def decide_opt_alloc(distrib):
    """
    Returns a list with the optimal allocation  of cache capacity (number of videos) allotted to each content provider
    
    Parameters
    ------------ 
    distrib: list
         list of lists of probabilities of requesting a video for each SP
   
    Uses global variables:
        
    nSP: int 
        number of SPs
    video_nb_list: list of int
        Number of videos provided by each SP
        
    Returns
    ----------
    allocation: list of ints
        list of the computed optimal cache capacity allocation for each content provider
    """
    
    allocation=[0]*nSP
    pointer_vec=[0]*nSP #point 
    if cache_capacity > sum(video_nb_list):
        return 'Error: Cache can contain all videos available, irrelevant for our project'
    for slot in range (cache_capacity):
# We keep one pointer per SP
# The pointer points to the first unconsumed object of each SP

# For each slot, we compare the popularity of the unconsumed objects of each SP
# We give this slot to the best SP, i.e., the one with the most popular unconsumed object
# The best SP "consumes" that object, i.e., it places it in the slot.
# Therefore, the pointer of the best SP moves to its next object.
        bestSP = 0 # Initialization

        for currentSP in range (1,nSP):

        	# We compare the popularity of the most popular unconsumed objects of bestSP and SP weighted by the probabilities of corresponding SP.
            bestSP_obj_popularity = distrib[bestSP][pointer_vec[bestSP]]*SP_proba[bestSP]
            currentSP_obj_popularity = distrib[currentSP][pointer_vec[currentSP]]*SP_proba[currentSP]
            if currentSP_obj_popularity > bestSP_obj_popularity:
                bestSP = currentSP

        # Now the bestSP has the most popular unconsumed object among SPs. We 
        # thus give it the slot.
        allocation[bestSP] = allocation[bestSP]+1

        # Now bestSP has consumed the object and we need to point to the next
        # object
        pointer_vec[bestSP] = pointer_vec[bestSP]+1
    return allocation




def evaluate_cost(allocation,first_alloc,best_alloc, requests_nb,video_probabi,bin_probabi):
    """
    Returns an estimation of the costs (number of requested videos that are not stored in the device and has to be transferred from server) 
    for three different allocations: Should be the current allocation, the first allocation (all slots allotted to last CP) and the computed optimal allocation
    
    Parameters
    ------------ 
    allocation: list of int
        Determined allocation
    first_alloc: list of int
        First allocation of the experiment (all slots allotted to last CP)
    best_alloc: list of int
        Optimal computed allocation
    requests_nb: int
        number of requests considered
    video_probabi: list of list of float
        list of lists of probabilities of requesting a video for each SP
    bin

    Returns
    ----------
    cost: int
        cost of allocation: number of requests non-satisfied by the cache 
    f_cost: int
        cost of first_alloc
    b_cost: int
        cost of best_alloc
    """   
    cost = 0
    b_cost=0
    f_cost=0
    for r in range(requests_nb):
        request = request_creation(video_probabi,bin_probabi)
        SP_of_the_video_requested = request[0]
        allocated_cache_space = allocation[SP_of_the_video_requested]
        f_allocated_cache_space = first_alloc[SP_of_the_video_requested]
        b_allocated_cache_space = best_alloc[SP_of_the_video_requested]
        video_id = request[1]
        if allocated_cache_space < video_id:
            cost +=1
        if b_allocated_cache_space < video_id:
            b_cost +=1
        if f_allocated_cache_space<video_id:
            f_cost+=1
    return(cost,f_cost,b_cost)

def states_nSP(capacity, numberSP, delta2): 
    """
    Returns all possibles states (cache capacity allocations) for n Content Providers, given a cache capacity (k) and a delta(pass of )
    
    Parameters
    ------------ 
    capacity: int
        The capacity of the cache
    numberSP: int
        Number of Service Providers
    delta2: int
        Number of slots by which we modify the storage allocation
         
    
    Returns
    ----------
    output_state_list: list of list of int
        The list with all possible states (all possible allocations).
        
    """   
    output_state_list=[]
    if numberSP==1:
        output_state_list=[[capacity]]
    elif numberSP==2:
        for j in range(int(capacity/delta2)+1):
            output_state_list.append([j*delta2, capacity-j*delta2])
        return(output_state_list)
    else:
        for i in range(int(capacity/delta2)+1): 
            other_states=states_nSP(capacity-i*delta2,numberSP-1,delta2)
            for state in other_states:
                state.append(i*delta2)
                output_state_list.append(state)
        return(output_state_list)
        
def get_state_index(alloc,delta):
    """
    Returns the state_index of a given allocation in the list of all possible states
    
    Parameters
    ------------ 
    alloc: list of int
        An allocation of cache capacity 
    delta: int
        Number of slots by which we modify the storage allocation 

    Returns
    ----------
    state_index: int
        the state_index of the allocation given in parameter
    """   
    
    k=sum(alloc)
    n=len(alloc)
    state_index=0
    for state in states_nSP(k,n,delta):
        if alloc==state:
            return state_index
        else:
            state_index+=1
            

            
def optimize_nSP(request_rate, nb_interval, interval_size, gama, epsi, alfa, delta, D, method): 
    """
    Algorithm with instantaneous reward equal to nominal_cost. Compute evolution of costs over the simulation
    (costs are normalized by the number of requests observed in an interval (interval_size*request_rate)
    Parameters
    ------------ 
    request_rate: int
        number of simulated requests per second
    nb_interval: int
        number of repetitions of the algorithm (number of actions taken)
    interval_size: int
        duration of an interval (time of observation before evaluating cost and taking new action)
    gama: float
        Usual gamma parameter of SARSA/Q-learning algorithm (discount factor)
    epsi: float
        Usual epsilon parameter of SARSA/Q-learning algorithm
    alfa: float
        Usual alpha parameter of SARSA/Q-learning algorithm (learning rate)
    delta: int
        Number of slots by which we modify the storage allocation
    D: set
        Possible action space of delta 
    method: string
        The optimization method (could be SARSA or Q-learning)
        
    Returns    
    ----------
    L_total_cost: list of float
        list containing  computed total cost evolution (nominal cost + perturbation cost) over intervals
    L_nominal_cost: list of float
        list containing computed nominal cost evolution over intervals
    L_first_cost: list of float
        list containing the evolution over interval of the cost of the initial allocation (independant of SARSA or Q-learning algorithm)
    L_best_cost: list of float
        list containing the evolution over interval of the cost of the optimal allocation (independant of SARSA or Q-learning algorithm)    
    """   

    init()
    (videos_proba,bins_proba)=catalog()
    L_total_cost=[]
    L_nominal_cost=[]
    L_first_cost=[]
    L_best_cost=[]
    request_nb = interval_size * request_rate #Number of requests in an interval
    best_allocation= decide_opt_alloc(videos_proba)
    #if DEBUG:
    #    print('best_allocation is',best_allocation)
    states = states_nSP(cache_capacity,nSP,delta)
    allocation =[0,0,1000] #first allocation: give all the cache to SP number 3
    first_allocation=list(allocation)
    state_index = get_state_index(allocation,delta)
    Q = np.zeros((nSP**2, len(states)))  
    V = np.zeros((nSP**2, len(states)))
    count_alea = 0 #number of exploratory actions
    count_best = 0 #number of exploitative actions
    D_size = len(D) #the size of actions space

    ###### ACTION %%%%%
    for j in range(nb_interval):
        alea = rd.random()
        old_allocation = deepcopy(allocation)
        coeff_ind = rd.randint(0, D_size-1)
        coeff = D[coeff_ind]
        print('iteration: ' + str(j) + ' ' + 'perturbation: ' + str(coeff * delta))
        if alea <= epsi: # epsilon-greedy policy
            action=rd.randint(0,nSP**2-1)
            action_plus = action//(nSP)
            action_minus = action % (nSP)
            allocation[action_plus] += coeff * delta
            allocation[action_minus] -= coeff * delta
            count_alea+=1
    
        else: 
            (best_score,best_actions) = af.find_max_list(Q[:, state_index])
            action=rd.choice(best_actions)
            action_plus=action // (nSP)
            action_minus=action % (nSP)
            allocation[action_plus] += coeff * delta
            allocation[action_minus] -= coeff * delta
        #if DEBUG:
        #    print('allocation is', allocation)
        #    print('action_plus is',action_plus)
        #    print('action_minus is',action_minus)
            count_best+=1
        allocation_ok=True
        for SP_cache in allocation:
            if  SP_cache<0 or SP_cache>cache_capacity:
                allocation = old_allocation
                Q[action][state_index] = 0
                allocation_ok=False
        #if DEBUG: 
        #    print('allocation exist',allocation_ok)
        if allocation_ok:
            #### REWARD COMPUTING #####
            (nominal_cost,first_cost,best_cost) = evaluate_cost(allocation,first_allocation,best_allocation,request_nb,videos_proba,bins_proba)
            perturbation_cost = (allocation[action_plus] - old_allocation[action_plus]) / request_nb
            nominal_cost = nominal_cost /request_nb
            total_cost = nominal_cost + perturbation_cost
            first_cost = first_cost /request_nb
            best_cost = best_cost /request_nb                
            new_gain = 1 - total_cost  #Reward
            #if DEBUG:
            #    print('new_gain is',new_gain)
            ## Q-TABLE UPDATE###
            state_index_prime = get_state_index(allocation,delta) #state_index of new state
            (best_score1,best_actions1)=af.find_max_list(Q[:,state_index_prime])
            #if DEBUG:
            #    print('allocation is ',allocation)
            #    print('action is ', action)
            if action_plus == action_minus:
                for act in range(nSP):
                    if method == 'Q_learning':
                        Q[act*nSP][state_index]+=alfa*(new_gain+gama*best_score1-Q[act][state_index]) #Q-Learning
                    if method == 'SARSA':
                        Q[act*nSP][state_index]+=alfa*(new_gain+gama*Q[act][state_index_prime]-Q[act][state_index]) #SARSA
                V[action][state_index]+=1
            else:
                if method == 'Q_learning':
                    Q[action][state_index]+=alfa*(new_gain+gama*best_score1-Q[action][state_index]) #Q-Learning
                if method == 'SARSA':
                    Q[action][state_index]+=alfa*(new_gain+gama*Q[action][state_index_prime]-Q[action][state_index]) #SARSA
            #if DEBUG:
            #    print('Q[action=',action,'],[state_index',state_index,']=',Q[action][state_index])
            #    print('Q[action=+',action_plus,'-',action_minus,'][,allocation', old_allocation,']=',Q[action][state_index])
            state_index = state_index_prime
            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
           # if DEBUG:
           #     print(total_cost)
           #     print(best_cost)
           #     print(Q)
           #     print(V)
           #     print(count_alea)
           #     print(count_best)
    return [L_total_cost,L_nominal_cost,L_first_cost,L_best_cost]
    
def write_results():
    """writes results of simulations in csv files
    """

    file = open("param.yml")
    parameters = yaml.load(file, Loader=yaml.FullLoader)
    request_rate = (parameters.get("request_rate"))[2] #0 for 10, 1 for 100 and 2 for 1000
    interval_size = (parameters.get("interval_size"))[0] #0 for 1 and 1 for 10
    delta1 = (parameters.get("delta1"))[1] #0 for 2, 1 for 10 and 2 for 50
    method = (parameters.get("method"))[1] #0 for SARSA, 1 for Q-learning and 2 for SPSA
    D = (parameters.get("D"))[2] #0 for [1], 1 for [1, 2, 4] and 2 for [1, 2, 4, 8, 16]

    nb_interval = int(40000/interval_size)
    [total_cost_sarsa,nominal_cost_sarsa,cost_first, cost_best] = optimize_nSP(request_rate, nb_interval, interval_size, gamma, epsilon, alpha_de_sarsa,delta1,D,method)
    filename = 'sarsa_cache1000_request_rate'+str(request_rate)+'_nb_interval'+str(nb_interval)+'interval_size'+str(interval_size)+'delta'+str(delta1)+'method_'+str(method)+'_coeffcients_'+str(D)+'.csv' # I changed the file name to clarify that we use the total cost
    f = open(filename, "w")
    f.write("Time_seconds,Time_hours,Total_Cost,Nominal_Cost,Cost_First,Best_Cost"+"\n")  
                
    for i in range(len(total_cost_sarsa)):
        f.write(str(i*interval_size)+","+str(i*interval_size/3600)+","+str(total_cost_sarsa[i])+","+str(nominal_cost_sarsa[i])+","+str(cost_first[i])+","+str(cost_best[i])+"\n")
                
    f.close()
    
write_results()