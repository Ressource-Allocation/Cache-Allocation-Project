#Fonctions utilisés pour uniquement 2 CPs
import CodeCassiopee as cc
import random as rd
import numpy as np


# Cette fonction complète la création d'un input
# Elle permet de créer une requête portant sur une vidéo i d'un content provider (Seulement Youtube ou Netflix)
def Request_creation(proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf):
    """ Cette fonction complète la création d'un input
        Elle permet de créer une requête portant sur une vidéo i d'un content provider"""
    Content_Provider = 'a determiner'
    distribution = []
    choix_CP = rd.random()
    if choix_CP <= proba_yt:
        Content_Provider = 'youtube'
        distribution = cc.zipf_distribution(alpha_yt, nb_videos_yt)
        choix_video_yt = rd.random()
        compteur_choix = 0
        for i in range(1, nb_videos_yt +1):
            compteur_choix += distribution[i-1]
            if compteur_choix >= choix_video_yt:
                video_choisie = i
                break        
    else:
        Content_Provider = 'netflix'
        distribution = cc.zipf_distribution(alpha_nf, nb_videos_nf)
        choix_video_nf = rd.random()
        compteur_choix = 0
        for i in range(1, nb_videos_nf +1):
            compteur_choix += distribution[i-1]
            if compteur_choix >= choix_video_nf:
                video_choisie = i
                break   
            
    return [Content_Provider, video_choisie]


#Permet d'évaluer le cout pour 2 CPs
def evaluate_cout_2(allocation, proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf, nb_requetes):
    cout = 0
    requete = Request_creation(proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf)
    if requete[0] == 'youtube':
        if requete[1] > allocation[0]:
            cout += 1
    else:
        if requete[1] > allocation[1]:
            cout += 1
    return cout



def sarsa_pour_2(intervalle, request_rate, proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf, cache_capacity):
    nb_iterations = intervalle * request_rate
    allocation = [cache_capacity/10.0 , 9*cache_capacity/10.0]
    index = allocation[0]
    rewards = np.zeros((3, 101))
    epsilon = rd.random()
    gain_init = nb_iterations - evaluate_cout_2(allocation, proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf, nb_iterations)
    for i in range (nb_iterations):
        alea = rd.random()
        if alea <= epsilon:
            action = rd.randint(-1,1)
            index += action
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
                position = 0
            elif action == -1:
                allocation[0] -= 1
                allocation[1] += 1
                position = 1
            elif action == 0:
                position = 2
        elif rewards[0, index] == rewards[1, index] and rewards[0, index] == rewards[2, index]:
            """TODO rewards [][] plutot non??"""
            action = rd.randint(-1,1)
            if action == 1:
                allocation[0] += 1
                allocation[1] -= 1
                position = 0
                index +=1
            elif action == -1:
                allocation[0] -= 1
                allocation[1] += 1
                position = 1
                index -= 1
            else:
                position = 2                
        elif rewards[0, index] == rewards[1, index]:
            if rewards[0, index] > rewards[2, index]:
                action = rd.randint(0,1)
                if action == 0:
                    allocation[0] += 1
                    allocation[1] -= 1
                    index += 1
                    position = 0
                else:
                    allocation[0] -= 1
                    allocation[1] += 1
                    index -= 1
                    position = 1
            else:
                action = 0
                position = 2
        elif rewards[0, index] == rewards[2, index]:
            if rewards[0, index] > rewards[1, index]:
                action = rd.randint(0,1)
                if action == 0:
                    allocation[0] += 1
                    allocation[1] -= 1
                    index += 1
                    position = 0
                else:
                    allocation[0] -= 0
                    allocation[1] += 0
                    position = 2
            else:
                allocation[0] -= 1
                allocation[1] += 1
                index -= 1
                position = 1
        elif rewards[1, index] == rewards[2, index]:
            if rewards[1, index] > rewards[0, index]:
                action = rd.randint(0,1)
                if action == 0:
                    allocation[0] -= 1
                    allocation[1] += 1
                    index -= 1
                    position = 1
                else:
                    allocation[0] -= 0
                    allocation[1] += 0
                    position = 2
            else:
                allocation[0] += 1
                allocation[1] -= 1
                index += 1
                position = 0
        else:
            max = rewards[:,index].max()
            if rewards[0, index] == max:
                position = 0
                index += 1
            if rewards[1, index] == max:
                position = 1
                index -=1
            if rewards[2, index] == max:
                position = 2
        nouv_gain = nb_iterations - evaluate_cout_2(allocation, proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf, nb_iterations)
        delta_gain = nouv_gain - gain_init
        gain_init = nouv_gain
        nouv_Q = fa.trouver_max_col(rewards, index)
        rewards[position][index] = delta_gain + nouv_Q[0]
    return allocation


        