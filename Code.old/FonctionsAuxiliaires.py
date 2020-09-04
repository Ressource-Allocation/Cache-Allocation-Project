#Liste des fonctions auxiliaires
import random as rd


#Calcule la somme d'une liste
def somme_liste(liste):
    somme=0
    for i in range(len(liste)):
        somme=somme+liste[i]
    return somme

        
def trouver_max_col(rewards, index):
    if rewards[0, index] == rewards[1, index] and rewards[0, index] == rewards[2, index]:
        action = rd.randint(-1,1)
        index += action
        if action == 1:
            position = 0
        elif action == -1:
            position = 1
        else:
            position = 2                
    elif rewards[0, index] == rewards[1, index]:
        if rewards[0, index] > rewards[2, index]:
            action = rd.randint(0,1)
            if action == 0:
                position = 0
            else:
                position = 1
        else:
            action = 0
            position = 2
    elif rewards[0, index] == rewards[2, index]:
        if rewards[0, index] > rewards[1, index]:
            action = rd.randint(0,1)
            if action == 0:
                position = 0
            else:
                position = 2
        else:
            position = 1
    elif rewards[1, index] == rewards[2, index]:
        if rewards[1, index] > rewards[0, index]:
            action = rd.randint(0,1)
            if action == 0:
                position = 1
            else:
                position = 2
        else:
            position = 0
    else:
        max = rewards[:,index].max()
        if rewards[0, index] == max:
            position = 0
        if rewards[1, index] == max:
            position = 1
        if rewards[2, index] == max:
            position = 2
    max = rewards[position, index]
    return [max, position, index]


#Renvoie l'indice du max d'un vecteur
def recherche_max(vec):   #on le fait sur une ligne de la matrice Q ie comme un vecteur vec
    max=vec[0]
    argmax=[]
    for i in range(len(vec)):  
        if vec[i]>max:
            max=vec[i]
            argmax=[i]
        elif vec[i]== max:
            argmax.append(i)
    choice=rd.choice([k for k in argmax])
    return choice







