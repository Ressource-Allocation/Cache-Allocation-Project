import matplotlib.pyplot as plt # necessaire pour tracer les plot
import random as rd
import numpy as np
import generator as gn # UTILISER : gn.fonction_machin
import FonctionsAuxiliaires as fa #Pour alléger le code principal
import time #Permet d'étudier les coûts de SARSA
from copy import deepcopy #permet de faire des copies de liste indépendante de l'originale

#Initialise les variables globales
def init():
    global liste_alpha #alpha correspondant aux fonctions de zipf pour chaque CP
    global liste_proba #Popularité de chaque CP
    global liste_nb_video #Nombre de vidéos proposé par chaque CP
    global cons_zipf_1 #alpha=0.8, 100 videos
    global cons_zipf_2 #alpha=1, 100 videos
    global cons_zipf_3 #alpha=1.2, 100 vidéos
    global conss_zipf #liste contenant les cons_zipf ci dessus
    global k #Nombre de CPs
    global cache_capacity #Taille du cache mémoire
    global alpha #alpha de zipf utilisé pour faire des tests
    global nb_videos #nb video pour chaque CP
    global gamma #paramétre de la l'équation pour calculer le nouveau Q dans SARSA
    global epsilon #Politique epsilon-greedy
    global alpha_de_sarsa #paramétre de la l'équation pour calculer le nouveau Q dans SARSA
    rd.seed(5)
    liste_alpha=[0.8, 1.0, 1.2]
    liste_proba=[0.7, 0.25, 0.05] #test pour essayer de trouver la convergence
    liste_nb_video=gn.liste_100_videos(k) #100 videos pour chaque CP
    cons_zipf_1 = 8.13443642804101 
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
    

    
# Cette fonction permet de créer un input sur les vidéos d'un content provider
# Elle retourne le graphe des probabilotés pi de la vidéo i en fonction de i
# nb_videos est le nombre de films du catalogue du content provider
# alpha est le paramètre présent dans la loi de distribution de zipf
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
    somme=0
    i=0
    choix_CP = rd.random()
    while(1==1):
        somme=somme+liste_proba[i]
        if(choix_CP <= somme):
            break
        else:
            i +=1
    CP=i
    distribution = zipf_distribution(liste_alpha[i], liste_nb_video[i], conss_zipf[i]) #conss_zipf permet de soulages les calculs des constantes de normalisation de zipf
    choix_video = rd.random()
    compteur_choix = 0
    for j in range(1, liste_nb_video[i]+1):
        compteur_choix += distribution[j-1]
        if compteur_choix >= choix_video:
            video_choisie = j
            break   
    return [CP, video_choisie]

    
# Cette fonction réalise une allocation naive du cache entre les content providers
def decide_naive_alloc(): # cache_capacity, k
    liste_nb_video=gn.liste_nb_de_video(k)
    liste_allocation=[0]*k
    nb_video_total=fa.somme_liste(liste_nb_video)
    for i in range(k):
        liste_allocation[i]=((1.0*liste_nb_video[i])/(nb_video_total))*cache_capacity
    return liste_allocation               
    

# Cette fonction réalise une allocation optimale du cache entre les content providers
# Le cache_capacity doit être inférieur au nombre total de vidéo (sinon pas de cassiopee)
def decide_opt_alloc(): # cache_capacity, k
    distribution=[0]*k
    popularite=[0]*k
    allocation=[0]*k
    pointeurs_max=[0]*k
    if cache_capacity > fa.somme_liste(liste_nb_video):
        return 'Erreur: Le cache est trop grand par rapport au nombre total de videos'
    for i in range(k):
        distribution[i]=zipf_distribution(liste_alpha[i], liste_nb_video[i], conss_zipf[i])
        popularite[i] = [piyt * liste_proba[i] for piyt in distribution[i]] #liste que l'on va comparer
    for j in range (cache_capacity + 1):
        max_temp=0 #popularite[0] par defaut
        for m in range(k-1):
            if popularite[m+1][pointeurs_max[m+1]]>popularite[m][pointeurs_max[m]]:
                max_temp=m+1
        allocation[max_temp]=allocation[max_temp] + 1
        pointeurs_max[max_temp] = pointeurs_max[max_temp] + 1 ;
    return allocation
     

# Cette fonction permet d'évaluer a posteriori le cout d'une allocation donnée
# La variable allocation est du type liste
def evaluate_cout(allocation, nb_requetes): # k
    cout = 0
    for r in range(1, nb_requetes +1):
        requete = request_creation() 
        if allocation[requete[0]]<requete[1]:
            cout +=1
    return cout



#Creations des 101 etats possibles pour 2 CPs
def states_2CP(cache_capacity): 
    liste=[]
    for i in range(cache_capacity+1):
        liste.append([i, cache_capacity-i])
    return liste
        

#Création des 5151 états possibles pour  3 CPs
def states_3CP(cache_capacity): 
    liste=[]
    for m in range (cache_capacity + 1): #le premier CP sur les 3
        for i in range(cache_capacity - m+1):
            liste.append([m, states_2CP(cache_capacity - m)[i][0], states_2CP(cache_capacity - m)[i][1]])
    return liste

        
#Recheche la position d'un état pour 3 CP
def position_etat(alloc): #la liste est en fait une allocation à 3 CP
    #la généralisation à k CP n'est pas faite ici par manque de la fonction states_kCP
    cache_capacity=alloc[0]+alloc[1]+alloc[2]
    compteur=-1
    for k in states_3CP(cache_capacity):
        compteur +=1 ;
        if alloc == k:
            return compteur


#Utile uniquement pour tester la convergence
def sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle): #intervalle, request_rate, gamma, epsilon, cache_capacity, alpha
    init()
    liste_cout=[]
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
            action=fa.recherche_max(Q[:, index])
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
            cout_1 = 0
            for cp in range(2):
                requetes_vers_le_cp = nb_iterations*liste_proba[cp] 
                hit_ratio = fa.somme_liste(zipf_distribution(liste_alpha[cp], nb_videos, conss_zipf[cp])[0 : (allocation[cp]-1)])
                cout_1 += requetes_vers_le_cp * (1- hit_ratio)
        else :
            cout_1 = evaluate_cout(allocation, nb_iterations)
        nouv_gain = nb_iterations - cout_1  # R dans la formule
        ## MISE A JOUR DE LA TABLE###
        index_prime = position_etat(allocation) # index du nouvel etat
        Q[action][index] = Q[action][index] + alpha_de_sarsa*(nouv_gain + gamma*Q[action][index_prime] - Q[action][index])
        index = index_prime
        liste_cout.append(cout_1)
    liste_cout_moyen=[]
    k=0
    while (k<nb_intervalle): #tous 100 intervalles
        liste_cout_moyen.append((fa.somme_liste(liste_cout[k : k+100]) / 100.0))
        k += 100
    plt.plot(range(len(liste_cout_moyen)), liste_cout_moyen, ".")
    #plt.xlim(4000, 5000)
    plt.title('Cout en fonction du nombre d\' itération' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Cout')
    plt.grid('on')
    #plt.rcParams["figure.figsize"] = [16, 9]
    plt.savefig('fig8.pdf')
    plt.close()
    plt.show() 
    return allocation


#Utile uniquement pour chercher les couts (temps de calcul) de differentes parties de l'algo
def sarsa_pour_3_bis(request_rate, intervalle): #intervalle, request_rate, gamma, epsilon, cache_capacity, alpha
    debut_algo=time.time()
    nb_iterations = intervalle * request_rate
    allocation = [int(10*cache_capacity/10) , int(0*cache_capacity/10), int(0*cache_capacity/10)]
    index = 0
    rewards = np.zeros((7, 5151))  
    gain_init = nb_iterations - evaluate_cout(allocation, nb_iterations)
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
            action=fa.recherche_max(rewards[:, index])
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
        cout_local = evaluate_cout(allocation, nb_iterations) #juste utilisé pour le print pour les tests
        nouv_gain = nb_iterations - cout_local
        delta_gain = nouv_gain - gain_init # R dans la formule
        gain_init = nouv_gain
        avant_Q = time.time() - debut_algo
        print('avant_Q : ', avant_Q)
        rewards[action][index] = rewards[action][index] + alpha_de_sarsa*(delta_gain + gamma*rewards[action][position_etat(allocation)] - rewards[action][index])
        apres_Q=time.time() - debut_algo
        print('apres changement Q : ', apres_Q)
        index = position_etat(allocation)
    fin=time.time() - debut_algo
    print('Temps total SARSA : ', fin)
    return allocation


#Utile uniquement pour tester les epsilon et gamma différents
def tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, gama, epsi, alfa): #intervalle, request_rate, gamma, epsilon, cache_capacity, alpha
    ### REQUEST RATE = 100 ###
    init()
    liste_cout=[]
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
            action=fa.recherche_max(Q[:, index])
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
            cout_1 = 0
            for cp in range(2):
                requetes_vers_le_cp = nb_iterations*liste_proba[cp] 
                hit_ratio = fa.somme_liste(zipf_distribution(liste_alpha[cp], nb_videos, conss_zipf[cp])[0 : (allocation[cp]-1)])
                cout_1 += requetes_vers_le_cp * (1- hit_ratio)
        else :
            cout_1 = evaluate_cout(allocation, nb_iterations)
        nouv_gain = nb_iterations - cout_1  # R dans la formule
        ## MISE A JOUR DE LA TABLE###
        index_prime = position_etat(allocation) # index du nouvel etat
        Q[action][index] = Q[action][index] + alfa*(nouv_gain + gama*Q[action][index_prime] - Q[action][index])
        index = index_prime
        liste_cout.append(cout_1)
    return liste_cout


def tests_de_gamma(request_rate, nb_intervalle, taille_intervalle):
    liste_gamma = [0.1, 0.3, 0.5, 0.7]
    for k in liste_gamma:
        cout_du_sarsa=tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, k, epsilon, alpha_de_sarsa)
        ## On fait une moyenne tous les 10 points intervalles
        liste_cout_moyen=[]
        i=0
        while (i<len(cout_du_sarsa)): #tous 10 intervalles
            liste_cout_moyen.append((fa.somme_liste(cout_du_sarsa[i : i+10]) / 10.0))
            i += 10
        plt.plot(range(len(liste_cout_moyen)), liste_cout_moyen, ".", label = str(k))
    plt.title('Test de Gamma' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Cout')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig('fig2.pdf')
    plt.close()
    plt.show()
       
    
def tests_de_epsilon(request_rate, nb_intervalle, taille_intervalle):
    liste_epsilon = [0.1, 0.3, 0.5, 0.9]
    for k in liste_epsilon:
        cout_du_sarsa=tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, gamma, k, alpha_de_sarsa)
        liste_cout_moyen=[]
        i=0
        while (i<len(cout_du_sarsa)): #tous 10 intervalles
            liste_cout_moyen.append((fa.somme_liste(cout_du_sarsa[i : i+10]) / 10.0))
            i += 10
        plt.plot(range(len(liste_cout_moyen)), liste_cout_moyen, ".", label = str(k))
    plt.title('Test de epsilon' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Cout')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig('fig8.pdf')
    plt.close()
    plt.show()
       
    
def tests_de_alpha(request_rate, nb_intervalle, taille_intervalle):
    liste_alpha = [0.1, 0.3, 0.5, 0.9]
    for k in liste_alpha:
        cout_du_sarsa=tests_sarsa_pour_3(request_rate, nb_intervalle, taille_intervalle, gamma, epsilon, k)
        liste_cout_moyen=[]
        i=0
        while (i<len(cout_du_sarsa)): #tous 10 intervalles
            liste_cout_moyen.append((fa.somme_liste(cout_du_sarsa[i : i+10]) / 10.0))
            i += 10
        plt.plot(range(len(liste_cout_moyen)), liste_cout_moyen, ".", label = str(k))
    plt.title('Test de alpha_de_sarsa' )
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Cout')
    plt.grid('on')
    plt.legend(loc = "best")
    plt.savefig('fig1.pdf')
    plt.close()
    plt.show()
    
       