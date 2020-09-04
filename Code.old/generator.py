import random as rd

#On crée la liste des k probas pour choisir tel ou tel CP
def liste_des_proba(k):
    liste_proba=[]
    somme=0
    for i in range(1, k+1):
        r=rd.random()
        somme=somme+r
        liste_proba.append(r)
    for j in range(0, k):
        liste_proba[j]=liste_proba[j]/somme
    return liste_proba
        

#Liste des probas uniforme
def liste_proba_uniforme(k):
    liste_proba=[]
    for i in range(k):
        liste_proba.append(1/k)
    return liste_proba


#On crée la liste des alphas pour les k CPs
def liste_des_alphas(k):
    liste_alpha=[]
    for i in range(k):
        liste_alpha.append(0.8)
    return liste_alpha


#On crée la liste des nombres de vidéos
def liste_nb_de_video(k):
    liste_video=[]
    for i in range(k):
        liste_video.append(1000)
    return liste_video


#liste des probas 
def liste_proba_seed(k):
    liste=[]
    somme=0
    for i in range(k):
        liste.append(rd.random())
        somme += liste[i]
    for p in liste:
        p /= somme  #normalisation pour somme=1
    return liste


#test creation liste des alpha avec seed(1) avec 3 valeurs de alphas (0.8, 1.0, 1.2)
def liste_alpha_seed1(k):
    rd.seed(1)
    l=[]
    for i in range(k):
        r=rd.random()
        if r<0.333333:
            l.append(0.8)
        elif r<0.666666:
            l.append(1.0)
        else:
            l.append(1.2)
    return l               #ca renvoie toujours la meme liste


#test creation liste des alpha avec seed(1) avec 3 valeurs de alphas (0.8, 1.0, 1.2)
def liste_alpha_seed2(k):
    rd.seed(2)
    l=[]
    for i in range(k):
        r=rd.random()
        if r<0.333333:
            l.append(0.8)
        elif r<0.666666:
            l.append(1.0)
        else:
            l.append(1.2)
    return l               #ca renvoie toujours la meme liste
        

#test creation liste des alpha avec seed(1) avec 3 valeurs de alphas (0.8, 1.0, 1.2)
def liste_alpha_seed3(k):
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
    return l               #ca renvoie toujours la meme liste


#Création liste 100 vidéo fix
def liste_100_videos(k):
    l=[]
    for i in range(k):
        l.append(100)
    return l
        

#Création liste 1000 vidéo fix
def liste_1000_videos(k):
    l=[]
    for i in range(k):
        l.append(1000)
    return l


#Création liste 10000 vidéo fix
def liste_10000_videos(k):
    l=[]
    for i in range(k):
        l.append(10000)
    return l