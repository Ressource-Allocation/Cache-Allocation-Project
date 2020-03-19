import matplotlib.pyplot as plt # necessaire pour tracer les plot (decocher si besoin)
import random as rd


def zipf_distribution(alpha, nb_videos): 
    """ Cette fonction permet de créer un input sur les vidéos d'un content provider
        Elle retourne le graphe des probabilotés pi de la vidéo i en fonction de i
        nb_videos est le nombre de films du catalogue du content provider
        alpha est le paramètre présent dans la loi de distribution de zipf"""
    norm=0
    indices_videos = range(1,nb_videos+1) # necessaire pour tracer le plot (decocher si besoin)
    probabilites_pi = [0] * (nb_videos)
    for i in range(1, nb_videos+1):
        norm +=1.0/(i**alpha)
    print(norm)
    for i in range(1, nb_videos+1):
        pi = (1.0/i**alpha) * (1.0/norm)
        probabilites_pi[i-1] = pi
               
    """plt.plot(indices_videos, probabilites_pi, "-")
    plt.title('Distribution Zipf')
    plt.xlabel('Indice des videos')
    plt.ylabel('Probabilites Zipf des videos')
    plt.grid('on')
    plt.show()"""
    return probabilites_pi
    

def Request_creation(proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf): #utiliser des listes de probas (de somme=1),d'alphas et nb_videos
    """ Cette fonction complète la création d'un input
        Elle permet de créer une requête portant sur une vidéo i d'un content provider"""
    Content_Provider = 'a determiner'
    distribution = []
    choix_CP = rd.random()
    if choix_CP <= proba_yt:
        Content_Provider = 'youtube'
        distribution = zipf_distribution(alpha_yt, nb_videos_yt)
        choix_video_yt = rd.random()
        compteur_choix = 0
        for i in range(1, nb_videos_yt +1):
            compteur_choix += distribution[i-1]
            if compteur_choix >= choix_video_yt:
                video_choisie = i
                break        
    else:
        Content_Provider = 'netflix'
        distribution = zipf_distribution(alpha_nf, nb_videos_nf)
        choix_video_nf = rd.random()
        compteur_choix = 0
        for i in range(1, nb_videos_nf +1):
            compteur_choix += distribution[i-1]
            if compteur_choix >= choix_video_nf:
                video_choisie = i
                break   
            
    return [Content_Provider, video_choisie]
    

def decide_naive_alloc(nb_videos_yt, nb_videos_nf, cache_capacity): #utiliser des listes de nombres de vidéos et retourner une liste d'allocation,
    """ Cette fonction réalise une allocation naive du cache entre les content providers"""
    allocation_yt = ((1.0 * nb_videos_yt)/(nb_videos_yt + nb_videos_nf)) * cache_capacity
    allocation_nf = ((1.0 * nb_videos_nf)/(nb_videos_yt + nb_videos_nf)) * cache_capacity
    return [allocation_yt, allocation_nf]
    

def decide_opt_alloc(proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf, cache_capacity):
    """ Cette fonction réalise une allocation optimale du cache entre les content providers
        Le cache_capacity doit être inférieur au nombre total de vidéo (sinon pas de cassiopee lol)"""
    if cache_capacity > nb_videos_yt + nb_videos_nf:
        return 'Erreur: Le cache est trop grand par rapport au nombre total de videos'
    distribution_yt = zipf_distribution(alpha_yt, nb_videos_yt)
    distribution_nf = zipf_distribution(alpha_nf, nb_videos_nf)
    popularite_yt = [piyt * proba_yt for piyt in distribution_yt]
    popularite_nf = [pinf * (1.0 - proba_yt) for pinf in distribution_nf]
    allocation_yt = 0
    allocation_nf = 0   
    
    for i in range(1, cache_capacity + 1):
        if max(popularite_yt) > max(popularite_nf):
            allocation_yt += 1
            popularite_yt.remove(max(popularite_yt))
            if len(popularite_yt) == 0:
                allocation_nf += cache_capacity - i
                break
        else:
            allocation_nf += 1
            popularite_nf.remove(max(popularite_nf))
            if len(popularite_nf) == 0:
                allocation_yt += cache_capacity -i
                break            
    
    return [allocation_yt, allocation_nf]
     

def evaluate_cout(allocation, proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf, nb_requetes):
    """ Cette fonction permet d'évaluer a posteriori le cout d'une allocation donnée
        La variable allocation est du type [allocation_yt , allocation_nf]"""
    cout = 0
    for r in range(1, nb_requetes +1):
        requete = Request_creation(proba_yt, alpha_yt, alpha_nf, nb_videos_yt, nb_videos_nf)
        if requete[0] == 'youtube':
            if requete[1] > allocation[0]:
                cout += 1
        else:
            if requete[1] > allocation[1]:
                cout += 1
    return cout, 1.0 *cout/nb_requetes
                
  
def main_tests():
    """Tests graphiques"""
    alpha1=0.6
    alpha2=1.5
    proba_yt=0.5
    nb_videos_yt=1000
    nb_videos_nf=1000
    cache_capacity=50
    nb_requetes=100
    
    """Test sur les alphas"""
    opt1_1=(evaluate_cout(decide_opt_alloc(proba_yt, alpha1, alpha2, nb_videos_yt, nb_videos_nf, cache_capacity), proba_yt, alpha1, alpha1, nb_videos_yt, nb_videos_nf, nb_requetes)[1])
    naive1_1=(evaluate_cout(decide_naive_alloc(nb_videos_yt, nb_videos_nf, cache_capacity), proba_yt, alpha1, alpha1, nb_videos_yt, nb_videos_nf, nb_requetes)[1])
    naive1_2=(evaluate_cout(decide_naive_alloc(nb_videos_yt, nb_videos_nf, cache_capacity), proba_yt, alpha1, alpha2, nb_videos_yt, nb_videos_nf, nb_requetes)[1])
    #print (naive1_1, naive1_2)
    plt.plot([0,1,1,0,0], [0, 0, opt1_1, opt1_1, 0], label="opt alpha=0.8" )
    plt.plot([1, 2, 2, 1, 1], [0, 0, naive1_1, naive1_1, 0], label=("naive alpha=0.8"))
    plt.plot([3,4,4,3,3], [0, 0, opt1_1, opt1_1, 0], label="opt alpha=0.8" )
    plt.plot([4,5,5,4,4], [0, 0, naive1_2, naive1_2, 0], label=("naive alpha=1.2"))
    plt.axis([-1, 6, 0, 1])
    plt.title("Comparaison des alphas")
    plt.legend(loc="best")
    plt.show()
    
    """Test sur le nombre de requete"""
    opt100=(evaluate_cout(decide_opt_alloc(proba_yt, alpha1, alpha1, nb_videos_yt, nb_videos_nf, cache_capacity), proba_yt, alpha1, alpha1, nb_videos_yt, nb_videos_nf, 100)[1])
    opt1000=(evaluate_cout(decide_opt_alloc(proba_yt, alpha1, alpha1, nb_videos_yt, nb_videos_nf, cache_capacity), proba_yt, alpha1, alpha1, nb_videos_yt, nb_videos_nf, 1000)[1])
    #print(opt100, opt1000)
    plt.plot([0,1,1,0,0], [0, 0, opt100, opt100, 0], label="opt 100 requete" )
    plt.plot([1, 2, 2, 1, 1], [0, 0, opt1000, opt1000, 0], label=("opt 1000 requete"))
    plt.axis([-0.5, 2.5, 0, 1])
    plt.title("Cout optimal en fonction du nombre de requete")
    plt.legend(loc="best")
    plt.show()
    
    """Test sur la proba de youtube"""
    opt0_4=(evaluate_cout(decide_opt_alloc(0.4, alpha1, alpha1, nb_videos_yt, nb_videos_nf, cache_capacity), 0.4, alpha1, alpha1, nb_videos_yt, nb_videos_nf, nb_requetes)[1])
    opt0_7=(evaluate_cout(decide_opt_alloc(0.7, alpha1, alpha1, nb_videos_yt, nb_videos_nf, cache_capacity), 0.7, alpha1, alpha1, nb_videos_yt, nb_videos_nf, nb_requetes)[1])
    #print(opt0_4, opt0_7)
    plt.plot([0,1,1,0,0], [0, 0, opt0_4, opt0_4, 0], label="opt proba you=0.4" )
    plt.plot([1, 2, 2, 1, 1], [0, 0,opt0_7,opt0_7, 0], label=("opt proba you=0.7"))
    plt.axis([-0.5, 2.5, 0, 1])
    plt.title("Cout optimal en fonction de la proba youtube")
    plt.legend(loc="best")
    plt.show()
    
    """Test sur les CP"""
    opt0_4=(evaluate_cout(decide_opt_alloc(0.4, alpha1, alpha1, nb_videos_yt, nb_videos_nf, cache_capacity), 0.4, alpha1, alpha1, nb_videos_yt, nb_videos_nf, nb_requetes)[1])
    opt0_7=(evaluate_cout(decide_opt_alloc(0.7, alpha1, alpha1, nb_videos_yt, nb_videos_nf, cache_capacity), 0.7, alpha1, alpha1, nb_videos_yt, nb_videos_nf, nb_requetes)[1])
    #print(opt0_4, opt0_7)
    plt.plot([0,1,1,0,0], [0, 0, opt0_4, opt0_4, 0], label="opt proba you=0.4" )
    plt.plot([1, 2, 2, 1, 1], [0, 0,opt0_7,opt0_7, 0], label=("opt proba you=0.7"))
    plt.axis([-0.5, 2.5, 0, 1])
    plt.title("Cout optimal en fonction de la proba youtube")
    plt.legend(loc="best")
    plt.show()
    
    
    
    
        
    
    
    
    



