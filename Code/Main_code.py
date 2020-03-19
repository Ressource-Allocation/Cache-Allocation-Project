#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:47:50 2019

@author: tdh
"""

import matplotlib.pyplot as plt
import random as rd
import numpy as np
import request_generator as gn # USE : gn.fonction_***
import Auxiliary_functions as af #To simplify code
import time #allow to compute SARSA complexity
from copy import deepcopy #allow to make independant copies of lists (not just aliases)

#Initialize global variables
def init():
    global list_alpha #alpha used in zipf functions for each CP
    global CP_proba #Popularity of each CP
    global video_nb_list #Number of videos provided by each CP
    global cons_zipf_1 #alpha=0.8, 100 videos
    global cons_zipf_2 #alpha=1, 100 videos
    global cons_zipf_3 #alpha=1.2, 100 vidéos
    global conss_zipf #list with cons_zipfs above
    global nCP #Number of CPs
    global cache_capacity #Cache capacity
    global alpha #zipf alpha use for tests
    global nb_videos #nb of videos for each CP
    global gamma #equation parameter to compute the new Q in SARSA
    global epsilon #epsilon-greedy politic
    global alpha_de_sarsa #equation parameter to compute the new Q in SARSA
    global W
    global DEBUG
    rd.seed(5)
    list_alpha=[0.8, 1.0, 1.2]
    CP_proba=[0.7, 0.25, 0.05] #test try and find convergence
    nCP=3
    video_nb_list=gn.list_100_videos(nCP) #100 videos by CP
    cons_zipf_1 = 8.13443642804101 #where do these numbers come from?
    cons_zipf_2 = 5.187377517639621
    cons_zipf_3 = 3.6030331432380347
    conss_zipf=[cons_zipf_1, cons_zipf_2, cons_zipf_3]
    cache_capacity =30
    alpha=0.8
    nb_videos = 100 
    gamma = 0.4
    epsilon = 0.5
    alpha_de_sarsa = 0.9
    W=1
    DEBUG=True

init() 
# Cette fonction permet de créer un input sur les vidéos d'un content provider
# Elle retourne le graphe des probabilotés pi de la vidéo i en fonction de i
# nb_videos est le nombre de films du catalogue du content provider
# alpha est le paramètre présent dans la loi de distribution de zipf
#A quoi correspond norme?
#compute_norm (calcul de la norme de zipf)
def zipf_distribution(alpha, nb_videos, norme):  
    "indices_videos = range(1,nb_videos+1)" # necessaire pour tracer le plot (decocher si besoin)
    probabilites_pi = [0] * (nb_videos)
    for i in range(1, nb_videos+1):
        pi = (1.0/i**alpha) * (1.0/norme)
        probabilites_pi[i-1] = pi          
    """
    liste_abscisse=[k for k in range(nb_videos)]
    plt.plot(liste_abscisse, probabilites_pi, "-")
    plt.title('Distribution Zipf (alpha=0.8, nb_videos=100)')
    plt.xlabel('Indice des videos')
    plt.ylabel('Probabilite de demande de la vidéo')
    plt.grid('on')
    plt.savefig('zipf_distribution.pdf')
    plt.close()
    plt.show()   
    """
    return probabilites_pi


# Cette fonction complète la création d'un input
# Elle permet de créer une requête portant sur une vidéo i d'un content provider
def request_creation(): #nCP
    """
    Returns the state_indexes of the selected content provider and the selected video in a simulated request.

    Parameters
    ------------ 
    Might be CP_proba, list_alpha, video_nb_list, conss_zipf, nCP
    
    Returns
    ----------
    [selected_CP, selected_video]
    selected_CP: int
        state_index of the selected content provider
    selected_video: int
        state_index of the selected video
    
    """
    S=CP_proba[0]
    selected_CP=0
    CP_choice = rd.random()
    while(CP_choice>S):
        selected_CP+=1
        S+=CP_proba[selected_CP]
    distribution = zipf_distribution(list_alpha[selected_CP], video_nb_list[selected_CP], conss_zipf[selected_CP]) #conss_zipf permet de soulages les calculs des constantes de normalisation de zipf
    video_choice = rd.random()
    S1 = distribution[0]
    selected_video=0
    while(video_choice>S1):
        selected_video+=1
        S1+=distribution[selected_video]
    return [selected_CP, selected_video]

    
def decide_naive_alloc(): # cache_capacity, nCP
    """
    Returns a list with a naive allocation of cache capacity (number of videos) for each content provider
    
    Parameters
    ------------ 
    Might be video_nb_list, k, cache_capacity
    
    Returns
    ----------
    list_allocation: list of ints
        list of naive cache capacity allocation for each content provider
    """
    video_nb_list=gn.video_nb_list(nCP)
    list_allocation=[0]*nCP
    nb_video_total=sum(video_nb_list)
    for i in range(nCP):
        list_allocation[i]=((1.0*video_nb_list[i])/(nb_video_total))*cache_capacity
    return list_allocation

         
# Cette fonction réalise une allocation optimale du cache entre les content providers
# Le cache_capacity doit être inférieur au nombre total de vidéo
def decide_opt_alloc():
    """
    Returns a list with the optimal allocation  of cache capacity (number of videos) for each content provider
    
    Parameters
    ------------ 
    Might be video_nb_list, k, cache_capacity, list_alpha, conss_zipf
    
    Returns
    ----------
    allocation: list of ints
        list of the optimal cache capacity allocation for each content provider
    """
    distribution=[0]*nCP
    popularity=[0]*nCP
    allocation=[0]*nCP
    pointer_vec=[0]*nCP #point 
    if cache_capacity > sum(video_nb_list):
        return 'Error: Cache can contain all videos available, irrelevant for our project'
    for i in range(nCP):
        distribution[i]=zipf_distribution(list_alpha[i], video_nb_list[i], conss_zipf[i])
        popularity[i] = [obj_proba_within_CP * CP_proba[i] for obj_proba_within_CP in distribution[i]] #list que l'on va comparer ??
    for slot in range (cache_capacity):
# We keep one pointer per CP
# The pointer points to the first unconsumed object of each CP

# For each slot, we compare the popularity of the unconsumed objects of each CP
# We give this slot to the best CP, i.e., the one with the most popular unconsumed object
# The best CP "consumes" that object, i.e., it places it in the slot.
# Therefore, the pointer of the best CP moves to its next object.
        bestCP = 0 # Initialization

        for currentCP in range (1,nCP):

        	# We compare the popularity of the most popular unconsumed objects 
        	# of bestCP and CP
            bestCP_obj_popularity = popularity[bestCP][pointer_vec[bestCP] ]
            currentCP_obj_popularity = popularity[currentCP][pointer_vec[currentCP] ]
            if currentCP_obj_popularity > bestCP_obj_popularity:
                bestCP = currentCP

        # Now the bestCP has the most popular unconsumed object among CPs. We 
        # thus give it the slot.
        allocation[bestCP] = allocation[bestCP]+1

        # Now bestCP has consumed the object and we need to point to the next
        # object
        pointer_vec[bestCP] = pointer_vec[bestCP]+1
    return allocation



# Cette fonction permet d'évaluer a posteriori le cost d'une allocation donnée
# La variable allocation est du type list
def evaluate_cost(allocation, requests_nb):
    """
    Returns an estimation of the cost (number of requested videos that are not stored in the device and has to be transferred from server) of a determined allocation
    
    Parameters
    ------------ 
    allocation: list of ints
        list of the optimal cache capacity allocation for each content provider
    requests_nb: int
        number of requests considered
    
    Returns
    ----------
    cost: int
        number of requested videos that are not stored in the device and has to be streamed from distant server
    """   
    cost = 0
    for r in range(requests_nb):
        request = request_creation()
        cp_of_the_video_requested = request[0]
        allocated_cache_space = allocation[cp_of_the_video_requested ]
        video_id = request[1]
        if allocated_cache_space < video_id:
            cost +=1
    return cost



#Creations des 101 etats possibles pour 2 CPs
def states_2CP(cache_capacity):
    """
    Returns all possibles states (cache capacity allocations) for 2 Content Providers, given a cache_capacity
    
    Parameters
    ------------ 
    cache_capacity: int
        The capacity of the cache 
    
    Returns
    ----------
    L: list of couples of int
        L is the list with all couples giving the cache capcity allocate to each Content Provider.
    """   
    L=[]
    for i in range(cache_capacity+1):
        L.append([i, cache_capacity-i])
    return L
        

#Création des 5151 états possibles pour  3 CPs
def states_3CP(cache_capacity): 
    """
    Returns all possibles states (cache capacity allocations) for 3 Content Providers, given a cache_capacity
    
    Parameters
    ------------ 
    cache_capacity: int
        The capacity of the cache 
    
    Returns
    ----------
    L: list of couples of int
        L is the list with all truples giving the cache capcity allocate to each Content Provider.
    """   
    L=[]
    for m in range (cache_capacity + 1): #le premier CP sur les 3
        for i in range(cache_capacity - m+1):
            L.append([m, states_2CP(cache_capacity - m)[i][0], states_2CP(cache_capacity - m)[i][1]])
    return L


def states_nCP(k,n):
    """
    Returns all possibles states (cache capacity allocations) for n Content Providers, given a cache capacity (k)
    
    Parameters
    ------------ 
    k: int
        The capacity of the cache
    n: int
        Number of Content Providers
    
    Returns
    ----------
    output_state_list: list of n_uples of int
        The list with all n_uples giving the cache capcity allocate to each Content Provider.
    """   
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
        
def get_state_index(alloc):
    """
    Returns the state_index of a given allocation in the list of all possible states
    
    Parameters
    ------------ 
    alloc: list of int
        an allocation of cache capacity 
    
    Returns
    ----------
    state_index: int
        the state_index of the allocation given in parameter
    """   
    
    k=sum(alloc)
    n=len(alloc)
    state_index=0
    for state in states_nCP(k,n):
        if alloc==state:
            return state_index
        else:
            state_index+=1
            
def sarsa_nCP(request_rate, nb_interval, interval_size):
    init()
    L_cost=[]
    if request_rate==-1:
        request_nb=1
    else:
        request_nb = interval_size * request_rate #Nombre de requête à chaque intervalle
    states=states_nCP(cache_capacity,nCP)
    allocation = rd.choice(states)
    state_index = get_state_index(allocation)
    Q = np.zeros((nCP*(nCP-1)+1, len(states)))  
    ###### ACTION ######♣
    for j in range(nb_interval):
        alea = rd.random()
        old_allocation=deepcopy(allocation)  #copie sur un autre pointeur
        if alea <= epsilon: #politique epsilon-greedy
            action=rd.randint(0,(nCP*(nCP-1)))
            action_plus = action//(nCP-1)
            action_minus=action % (nCP-1)
            if action_plus!= nCP:
                allocation[action_plus]+=1
                allocation[action_minus]-=1
                
        else: #on cherche le max
            (best_score,best_actions)=af.find_max_list(Q[:, state_index])
            action=rd.choice(best_actions)
            action_plus=action // (nCP-1)
            action_minus=action % (nCP-1)
            if action_plus != nCP:
                allocation[action_plus]+=1
                allocation[action_minus]-=1

                

        if -1 in allocation:
            allocation = old_allocation
            Q[action][state_index] = 0
        #### CALCUL DU GAIN #####
        if request_rate == -1:
            cost_1 = 0
            for cp in range(nCP):
                requests_to_cp = request_nb*CP_proba[cp] 
                if allocation[cp]==0:
                    hit_ratio=0
                else:
                    hit_ratio = af.list_sum(zipf_distribution(list_alpha[cp], nb_videos,conss_zipf[cp])[0 : (allocation[cp])])
                cost_1 += requests_to_cp * (1- hit_ratio)
        else :
            cost_1 = evaluate_cost(allocation, request_nb)+allocation[action_plus]-old_allocation[action_plus]
        new_gain = request_nb - cost_1  # R dans la formule
        ## MISE A JOUR DE LA TABLE###
        state_index_prime = get_state_index(allocation) # state_index du nouvel etat
        Q[action][state_index] = Q[action][state_index] + alpha_de_sarsa*(new_gain + gamma*Q[action][state_index_prime] - Q[action][state_index])
        state_index = state_index_prime
        L_cost.append(cost_1)
    L_cost_mean=[]
    k=0
    while (k<nb_interval): #tous 100 intervalles
        L_cost_mean.append((af.list_sum(L_cost[k : k+W]) /W))
        k +=W
    filename= 'sarsa_nCP_request_rate'+str(request_rate)+'_nb_interval'+str(nb_interval)+'interval_size'+str(interval_size)+'.pdf'
    plt.plot(range(len(L_cost_mean)), L_cost_mean, ".")
    #plt.xlim(0, 1000)
    plt.title('cost en fonction du nombre d\' itération' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('cost')
    plt.grid('on')
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.savefig(filename)
    #plt.close()
    plt.show() 
    return allocation
           
            

def test_sarsa_nCP(request_rate, nb_interval, interval_size, gama, epsi, alfa,delta): #intervalle, request_rate, gamma, epsilon, cache_capacity, alpha
    ### REQUEST RATE = 100 ###
    init()
    L_cost=[]
    L_best_cost=[]
    if request_rate == -1:
    ### CAS THEORIQUE : NOMBRE DE REQUETES INFINI ####
        request_nb=1
    else:
        request_nb = interval_size * request_rate #Nombre de requête à chaque intervalle
    best_allocation= decide_opt_alloc()
    states=states_nCP(cache_capacity,nCP)
    allocation = rd.choice(states)
    #allocation=[cache_capacity, 0,0]
    state_index = get_state_index(allocation)
    Q = np.zeros((nCP**2, len(states)))
    V = np.zeros((nCP**2, len(states)))
    count_alea=0 #number of random actions
    count_best=0
    ###### ACTION %%%%%
    for j in range(nb_interval):
        alea = rd.random()
        old_allocation=deepcopy(allocation)  #copie sur un autre pointeur
        if alea <= epsi: #politique epsilon-greedy
            action=rd.randint(0,nCP**2-1)
            action_plus = action//(nCP)
            action_minus=action % (nCP)
            allocation[action_plus]+=1
            allocation[action_minus]-=1
            count_alea+=1
    
        else: #on cherche le max
            (best_score,best_actions)=af.find_max_list(Q[:, state_index])
            action=rd.choice(best_actions)
            action_plus=action // (nCP)
            action_minus=action % (nCP)
            allocation[action_plus]+=1
            allocation[action_minus]-=1
            count_best+=1
        if -1 in allocation:
            allocation = old_allocation
            Q[action][state_index] = 0

        else:
            #### CALCUL DU GAIN #####
            if request_rate == -1:
            # In this case we do not really generate any requests
            # but we just compute the probability of finding the video
            # in the cache. In this case the cost is represented by this
            # probability
                cost_1 = 0
                best_cost=0
                for cp in range(nCP):
                    probability_of_requesting_cp = CP_proba[cp]
                    hit_ratio = af.list_sum(zipf_distribution(list_alpha[cp], nb_videos, conss_zipf[cp])[0 :allocation[cp]])
                    hit_ratio_best=  af.list_sum(zipf_distribution(list_alpha[cp], nb_videos, conss_zipf[cp])[0 :best_allocation[cp]])
                    cost_1 +=probability_of_requesting_cp*(1- hit_ratio)
                    best_cost+=probability_of_requesting_cp*(1- hit_ratio_best)
                    if DEBUG:
                        print('hit_ratio is',hit_ratio)
                        print('cost_1 is',cost_1)
            else :
                cost_1 =(evaluate_cost(allocation, request_nb)+allocation[action_plus]-old_allocation[action_plus])/request_nb
                best_cost= evaluate_cost(best_allocation,request_nb)/request_nb
                
            new_gain =1-cost_1  # R dans la formule
            if DEBUG:
                print('new_gain is',new_gain)
            ## MISE A JOUR DE LA TABLE###
            state_index_prime = get_state_index(allocation) # state_index du nouvel etat
            (best_score1,best_actions1)=af.find_max_list(Q[:,state_index_prime])
            if DEBUG:
                print('allocation is ',allocation)
                print('action is ', action)
            if action_plus==action_minus:
                for act in range(nCP):
                    Q[act*nCP][state_index]+=alfa*(new_gain+gama*best_score1-Q[act][state_index])
                    #Q[act*nCP][state_index]+= alfa*(new_gain + gama*Q[act][state_index_prime] - Q[act][state_index])
                V[action][state_index]+=1
            else:
               Q[action][state_index]+=alfa*(new_gain+gama*best_score1-Q[action][state_index])
                    #Q[action*nCP][state_index]+= alfa*(new_gain + gama*Q[action][state_index_prime] - Q[action][state_index])
            #if DEBUG:
                #print('Q[action=',action,'],[state_index',state_index,']=',Q[action][state_index])
                #print('Q[action=+',action_plus,'-',action_minus,'][,allocation', old_allocation,']=',Q[action][state_index])
            state_index = state_index_prime
            L_cost.append(cost_1)
            L_best_cost.append(best_cost)
            if DEBUG:
                print(cost_1)
                print(best_cost)
    return [L_cost,L_best_cost,count_alea,count_best,Q,V]


def tests_gamma(request_rate, nb_interval, interval_size):
    L_gamma = [0.1, 0.3, 0.5, 0.7]
    for k in L_gamma:
        cost_sarsa=test_sarsa_nCP(request_rate, nb_interval, interval_size, k, epsilon, alpha_de_sarsa)
        L_cost_mean=[]
        i=0
        while (i<len(cost_sarsa)): #tous 10 intervalles
            # Instead of representing the cost per each iteration, we plot 
            # the average of the cost over W iterations
            L_cost_mean.append((af.list_sum(cost_sarsa[i : i+W]) / W))
            i += W
        plt.plot(range(len(L_cost_mean)), L_cost_mean, ".", label = str(k))
    filename= 'test_de_gamma_request_rate'+str(request_rate)+'_nb_interval'+str(nb_interval)+'interval_size'+str(interval_size)+'.pdf'
    plt.title('Test de Gamma' )
    plt.xlabel('Nombre de groupe de W itérations')
    plt.ylabel('cost')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig(filename)
    #plt.close()
    plt.show()
       
    
def tests_epsilon(request_rate, nb_interval, interval_size):
    for epsi1 in [0.1,0.2,0.3]:
        [cost_sarsa,cost_best,count_alea,count_best,Qtable,Vtable]=test_sarsa_nCP(request_rate, nb_interval, interval_size, gamma, epsi1, alpha_de_sarsa)
        L_cost_mean=[]
        L_best_cost_mean=[]
        i=0
        while (i<len(cost_sarsa)-W): #tous 10 intervalles
            L_cost_mean.append((af.list_sum(cost_sarsa[i : i+W]) /W))
            L_best_cost_mean.append((af.list_sum(cost_best[i:i+W])/W))
            #L_time.append()
            i += W
            if DEBUG:
                print('count_alea is ',count_alea,' count_best is ',count_best)
                print(Qtable)
                print(Vtable)
        plt.plot(range(len(L_cost_mean)), L_cost_mean, ".", label = str(epsi1))
        plt.plot(range(len(L_best_cost_mean)), L_best_cost_mean, ".", label ='best')
        filename= 'test_de_epsilon'+str(epsi1)+'_request_rate'+str(request_rate)+'_nb_interval'+str(nb_interval)+'interval_size'+str(interval_size)+'.pdf'
        plt.title('Test de epsilon' )
        plt.xlabel('Time')
        plt.ylabel('cost')
        plt.grid('on')
        plt.legend(loc = "best")
        plt.draw()
        plt.savefig(filename)
        #plt.show()

    
def tests_alpha(request_rate, nb_interval, interval_size):
    list_alpha = [0.1, 0.3, 0.5, 0.9]
    for k in list_alpha:
        cost_sarsa=test_sarsa_nCP(request_rate, nb_interval, interval_size, gamma, epsilon, k)
        L_cost_mean=[]
        i=0
        while (i<len(cost_sarsa)): #tous 10 intervalles
            L_cost_mean.append((af.list_sum(cost_sarsa[i : i+10]) / 10.0))
            i += 10
        plt.plot(range(len(L_cost_mean)), L_cost_mean, ".", label = str(k))
    filename= 'test_de_alpha_request_rate'+str(request_rate)+'_nb_interval'+str(nb_interval)+'interval_size'+str(interval_size)+'.pdf'
    plt.title('Test de alpha_de_sarsa' )
    plt.xlabel('Interval number')
    plt.ylabel('cost')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig(filename)
    plt.show()
    
def tests_for_ploting(fac):
    for i in range(0,5):
        for j in range (0,5):
            for k in range(0,5):
                
                sarsa_nCP(fac*i+1,fac*j+1,fac*k+1)
                tests_gamma(fac*i+1,fac*j+1,fac*k+1)
                tests_epsilon(fac*i+1,fac*j+1,fac*k+1)
                tests_alpha(fac*i+1,fac*j+1,fac*k+1)



    
       