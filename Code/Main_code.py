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
    global list_proba #Popularity of each CP
    global video_nb_list #Number of videos provided by each CP
    global cons_zipf_1 #alpha=0.8, 100 videos
    global cons_zipf_2 #alpha=1, 100 videos
    global cons_zipf_3 #alpha=1.2, 100 vidéos
    global conss_zipf #list with cons_zipfs above
    global k #Number of CPs
    global cache_capacity #Cache capacity
    global alpha #zipf alpha use for tests
    global nb_videos #nb of videos for each CP
    global gamma #equation parameter to compute the new Q in SARSA
    global epsilon #epsilon-greedy politic
    global alpha_de_sarsa #equation parameter to compute the new Q in SARSA
    rd.seed(5)
    list_alpha=[0.8, 1.0, 1.2]
    list_proba=[0.7, 0.25, 0.05] #test try and find convergence
    video_nb_list=gn.list_100_videos(k) #100 videos by CP
    cons_zipf_1 = 8.13443642804101 #where do these numbers come from?
    cons_zipf_2 = 5.187377517639621
    cons_zipf_3 = 3.6030331432380347
    conss_zipf=[cons_zipf_1, cons_zipf_2, cons_zipf_3]
    k=3  
    cache_capacity = 30
    alpha=0.8
    nb_videos = 100 
    gamma = 0.8
    epsilon = 0.5
    alpha_de_sarsa = 0.9
    

init() 
# Cette fonction permet de créer un input sur les vidéos d'un content provider
# Elle retourne le graphe des probabilotés pi de la vidéo i en fonction de i
# nb_videos est le nombre de films du catalogue du content provider
# alpha est le paramètre présent dans la loi de distribution de zipf
#A quoi correspond norme?
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
def request_creation(): #k
    """
    Returns the indexes of the selected content provider and the selected video in a simulated request.

    Parameters
    ------------ 
    Might be list_proba, list_alpha, video_nb_list, conss_zipf, k
    
    Returns
    ----------
    [selected_CP, selected_video]
    selected_CP: int
        index of the selected content provider
    selected_video: int
        index of the selected video
    
    """
    S=0
    selected_CP=-1
    CP_choice = rd.random()
    while(CP_choice>S):
        S+=list_proba[selected_CP]
        selected_CP+=1
    if selected_CP<0:
        selected_CP+=1
    distribution = zipf_distribution(list_alpha[selected_CP], video_nb_list[selected_CP], conss_zipf[selected_CP]) #conss_zipf permet de soulages les calculs des constantes de normalisation de zipf
    video_choice = rd.random()
    S1 = 0
    selected_video=-1
    while(video_choice>S1):
        S1+=distribution[selected_video]
        selected_video+=1
    if selected_video<0:
        selected_video+=1
    return [selected_CP, selected_video]

    
def decide_naive_alloc(): # cache_capacity, k
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
    video_nb_list=gn.video_nb_list(k)
    list_allocation=[0]*k
    nb_video_total=sum(video_nb_list)
    for i in range(k):
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
    distribution=[0]*k
    popularity=[0]*k
    allocation=[0]*k
    pointers_max=[0]*k
    if cache_capacity > sum(video_nb_list):
        return 'Error: Cache can contain all videos available, irrelevant for our project'
    for i in range(k):
        distribution[i]=zipf_distribution(list_alpha[i], video_nb_list[i], conss_zipf[i])
        popularity[i] = [piyt * list_proba[i] for piyt in distribution[i]] #list que l'on va comparer ??
    for j in range (cache_capacity):
        max_temp=0 # default popularity[0] 
        for m in range(k-1):
            if popularity[m+1][pointers_max[m+1]]>popularity[m][pointers_max[m]]:
                max_temp=m+1
        allocation[max_temp]=allocation[max_temp]+ 1
        pointers_max[max_temp] = pointers_max[max_temp] + 1 ;
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
        if allocation[request[0]]<request[1]:
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
        
def state_index(alloc):
    """
    Returns the index of a given allocation in the list of all possible states
    
    Parameters
    ------------ 
    alloc: list of int
        an allocation of cache capacity 
    
    Returns
    ----------
    index: int
        the index of the allocation given in parameter
    """   
    
    k=sum(alloc)
    n=len(alloc)
    index=0
    for state in states_nCP(k,n):
        if alloc==state:
            return index
        else:
            index+=1
            
            
            
        
#Utile uniquement pour tester la convergence
def sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle): #intervalle, request_rate, gamma, epsilon, cache_capacity, alpha
    init()
    L_cost=[]
    ### CAS THEORIQUE : NOMBRE DE REQUETES INFINI ####
    if request_rate == -1:
        nb_iterations = 100
    else:
        nb_iterations = taille_intervalle * request_rate #Nombre de requête à chaque intervalle
    allocation = [int(0*cache_capacity/10) , int(0*cache_capacity/10), int(10*cache_capacity/10)]
    index = 0
    Q = np.zeros((7, 5151))  
    ###### ACTION ######♣
    for j in range(nb_intervalle):
        alea = rd.random()
        old_allocation=deepcopy(allocation)  #copie sur un autre pointeur
        if alea <= epsilon: #politique epsilon-greedy
            action = rd.randint(0,6) #random entre 0 et 6 inclus --> 7 actions possibles
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
            if action == 2:
                allocation[0] -= 1
                allocation[1] += 1
            if action == 3:
                allocation[0] += 1
                allocation[2] -= 1
            if action == 4:
                allocation[0] -= 1
                allocation[2] += 1
            if action == 5:
                allocation[1] += 1
                allocation[2] -= 1
            if action == 6:
                allocation[1] -= 1
                allocation[2] += 1
        else: #on cherche le max
            action=af.recherche_max(Q[:, index])
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
            if action == 2:
                allocation[0] -= 1
                allocation[1] += 1
            if action == 3:
                allocation[0] += 1
                allocation[2] -= 1
            if action == 4:
                allocation[0] -= 1
                allocation[2] += 1
            if action == 5:
                allocation[1] += 1
                allocation[2] -= 1
            if action == 6:
                allocation[1] -= 1
                allocation[2] += 1  
        if -1 in allocation:
            allocation = old_allocation
            Q[action][index] = 0
        #### CALCUL DU GAIN #####
        if request_rate == -1:
            cost_1 = 0
            for cp in range(2):
                requetes_vers_le_cp = nb_iterations*list_proba[cp] 
                hit_ratio = af.list_sum(zipf_distribution(list_alpha[cp], nb_videos, conss_zipf[cp])[0 : (allocation[cp]-1)])
                cost_1 += requetes_vers_le_cp * (1- hit_ratio)
        else :
            cost_1 = evaluate_cost(allocation, nb_iterations)
        nouv_gain = nb_iterations - cost_1  # R dans la formule
        ## MISE A JOUR DE LA TABLE###
        index_prime = state_index(allocation) # index du nouvel etat
        Q[action][index] = Q[action][index] + alpha_de_sarsa*(nouv_gain + gamma*Q[action][index_prime] - Q[action][index])
        index = index_prime
        L_cost.append(cost_1)
    L_cost_moyen=[]
    k=0
    while (k<nb_intervalle): #tous 100 intervalles
        L_cost_moyen.append((af.list_sum(L_cost[k : k+100]) / 100.0))
        k += 100
    plt.plot(range(len(L_cost_moyen)), L_cost_moyen, ".")
    #plt.xlim(4000, 5000)
    plt.title('cost en fonction du nombre d\' itération' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('cost')
    plt.grid('on')
    #plt.rcParams["figure.figsize"] = [16, 9]
    plt.savefig('fig8.pdf')
    plt.close()
    plt.show() 
    return allocation


#Utile uniquement pour chercher les costs (temps de calcul) de differentes parties de l'algo
def sarsa_pour_3_bis(request_rate, intervalle): #intervalle, request_rate, gamma, epsilon, cache_capacity, alpha
    debut_algo=time.time()
    nb_iterations = intervalle * request_rate
    allocation = [int(10*cache_capacity/10) , int(0*cache_capacity/10), int(0*cache_capacity/10)]
    index = 0
    rewards = np.zeros((7, 5151))  
    gain_init = nb_iterations - evaluate_cost(allocation, nb_iterations)
    for i in range (0, nb_iterations):  #nb de requete
        alea = rd.random()
        old_allocation=deepcopy(allocation)  #copie sur un autre pointeur
        if alea <= epsilon: #politique epsilon-greedy
            action = rd.randint(0,6) #random entre 0 et 6 inclus --> 7 actions possibles
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
            if action == 2:
                allocation[0] -= 1
                allocation[1] += 1
            if action == 3:
                allocation[0] += 1
                allocation[2] -= 1
            if action == 4:
                allocation[0] -= 1
                allocation[2] += 1
            if action == 5:
                allocation[1] += 1
                allocation[2] -= 1
            if action == 6:
                allocation[1] -= 1
                allocation[2] += 1
        else: #on cherche le max
            action=af.recherche_max(rewards[:, index])
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
            if action == 2:
                allocation[0] -= 1
                allocation[1] += 1
            if action == 3:
                allocation[0] += 1
                allocation[2] -= 1
            if action == 4:
                allocation[0] -= 1
                allocation[2] += 1
            if action == 5:
                allocation[1] += 1
                allocation[2] -= 1
            if action == 6:
                allocation[1] -= 1
                allocation[2] += 1  
        if -1 in allocation:
            allocation = old_allocation
            rewards[action][index] = -150000000.0
        cost_local = evaluate_cost(allocation, nb_iterations) #juste utilisé pour le print pour les tests
        nouv_gain = nb_iterations - cost_local
        delta_gain = nouv_gain - gain_init # R dans la formule
        gain_init = nouv_gain
        avant_Q = time.time() - debut_algo
        print('avant_Q : ', avant_Q)
        rewards[action][index] = rewards[action][index] + alpha_de_sarsa*(delta_gain + gamma*rewards[action][state_index(allocation)] - rewards[action][index])
        apres_Q=time.time() - debut_algo
        print('apres changement Q : ', apres_Q)
        index = state_index(allocation)
    fin=time.time() - debut_algo
    print('Temps total SARSA : ', fin)
    return allocation


#Utile uniquement pour tester les epsilon et gamma différents
def tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, gama, epsi, alfa): #intervalle, request_rate, gamma, epsilon, cache_capacity, alpha
    ### REQUEST RATE = 100 ###
    init()
    L_cost=[]
    ### CAS THEORIQUE : NOMBRE DE REQUETES INFINI ####
    if request_rate == -1:
        nb_iterations = 100
    else:
        nb_iterations = taille_intervalle * request_rate #Nombre de requête à chaque intervalle
    allocation = [int(0*cache_capacity/10) , int(0*cache_capacity/10), int(10*cache_capacity/10)]
    index = 0
    Q = np.zeros((7, 5151))  
    ###### ACTION %%%%%
    for j in range(nb_intervalle):
        alea = rd.random()
        old_allocation=deepcopy(allocation)  #copie sur un autre pointeur
        if alea <= epsi: #politique epsilon-greedy
            action = rd.randint(0,6) #random entre 0 et 6 inclus --> 7 actions possibles
            #position=action
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
            if action == 2:
                allocation[0] -= 1
                allocation[1] += 1
            if action == 3:
                allocation[0] += 1
                allocation[2] -= 1
            if action == 4:
                allocation[0] -= 1
                allocation[2] += 1
            if action == 5:
                allocation[1] += 1
                allocation[2] -= 1
            if action == 6:
                allocation[1] -= 1
                allocation[2] += 1
        else: #on cherche le max
            action=af.recherche_max(Q[:, index])
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
            if action == 2:
                allocation[0] -= 1
                allocation[1] += 1
            if action == 3:
                allocation[0] += 1
                allocation[2] -= 1
            if action == 4:
                allocation[0] -= 1
                allocation[2] += 1
            if action == 5:
                allocation[1] += 1
                allocation[2] -= 1
            if action == 6:
                allocation[1] -= 1
                allocation[2] += 1  
        if -1 in allocation:
            allocation = old_allocation
            Q[action][index] = 0
        #### CALCUL DU GAIN #####
        if request_rate == -1:
            cost_1 = 0
            for cp in range(2):
                requetes_vers_le_cp = nb_iterations*list_proba[cp] 
                hit_ratio = af.list_sum(zipf_distribution(list_alpha[cp], nb_videos, conss_zipf[cp])[0 : (allocation[cp]-1)])
                cost_1 += requetes_vers_le_cp * (1- hit_ratio)
        else :
            cost_1 = evaluate_cost(allocation, nb_iterations)
        nouv_gain = nb_iterations - cost_1  # R dans la formule
        ## MISE A JOUR DE LA TABLE###
        index_prime = state_index(allocation) # index du nouvel etat
        Q[action][index] = Q[action][index] + alfa*(nouv_gain + gama*Q[action][index_prime] - Q[action][index])
        index = index_prime
        L_cost.append(cost_1)
    return L_cost


def tests_de_gamma(request_rate, nb_intervalle, taille_intervalle):
    L_gamma = [0.1, 0.3, 0.5, 0.7]
    for k in L_gamma:
        cost_du_sarsa=tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, k, epsilon, alpha_de_sarsa)
        ## On fait une moyenne tous les 10 points intervalles
        L_cost_moyen=[]
        i=0
        while (i<len(cost_du_sarsa)): #tous 10 intervalles
            L_cost_moyen.append((af.list_sum(cost_du_sarsa[i : i+10]) / 10.0))
            i += 10
        plt.plot(range(len(L_cost_moyen)), L_cost_moyen, ".", label = str(k))
    plt.title('Test de Gamma' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('cost')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig('fig2.pdf')
    plt.close()
    plt.show()
       
    
def tests_de_epsilon(request_rate, nb_intervalle, taille_intervalle):
    L_epsilon = [0.1, 0.3, 0.5, 0.9]
    for k in L_epsilon:
        cost_du_sarsa=tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, gamma, k, alpha_de_sarsa)
        L_cost_moyen=[]
        i=0
        while (i<len(cost_du_sarsa)): #tous 10 intervalles
            L_cost_moyen.append((af.list_sum(cost_du_sarsa[i : i+10]) / 10.0))
            i += 10
        plt.plot(range(len(L_cost_moyen)), L_cost_moyen, ".", label = str(k))
    plt.title('Test de epsilon' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('cost')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig('fig8.pdf')
    plt.close()
    plt.show()
       
    
def tests_de_alpha(request_rate, nb_intervalle, taille_intervalle):
    list_alpha = [0.1, 0.3, 0.5, 0.9]
    for k in list_alpha:
        cost_du_sarsa=tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, gamma, epsilon, k)
        L_cost_moyen=[]
        i=0
        while (i<len(cost_du_sarsa)): #tous 10 intervalles
            L_cost_moyen.append((af.list_sum(cost_du_sarsa[i : i+10]) / 10.0))
            i += 10
        plt.plot(range(len(L_cost_moyen)), L_cost_moyen, ".", label = str(k))
    plt.title('Test de alpha_de_sarsa' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('cost')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig('fig1.pdf')
    plt.close()
    plt.show()
    
       