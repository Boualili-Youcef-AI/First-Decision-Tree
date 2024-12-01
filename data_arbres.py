import numpy as np
import math

class DataPoint:
    # x c'est les attributs 
    # y c'est les classes
    # cles c'est le nom des attributs
    def __init__(self, x, y, cles):
        self.x = {} # dic
        for i in range(len(cles)):
            self.x[cles[i]] = float(x[i])
        self.y = int(y)
        self.dim = len(self.x)
        
    def __repr__(self):
        return 'x: '+str(self.x)+', y: '+str(self.y)

#point = DataPoint([1.2, 3.4, 5.6], 1, ["taille", "poids", "âge"])


def load_data(filelocation):
    with open(filelocation,'r') as f:
        data = []
        attributs = f.readline()[:-1].split(',')[:-1]
        for line in f:
            z = line.split(',')
            if z[-1] == '\n':
                z = z[:-1]
            x = z[:-1]
            y = int(z[-1])
            data.append(DataPoint(x,y,attributs))
    return data


class Noeud:
    def __init__(self, profondeur_max=np.infty):
        self.question = None
        self.enfants = {}
        self.profondeur_max = profondeur_max
        self.proba = None

    def prediction(self, x):
        pass
        
    def grow(self, data):
        pass

# Calcul de la probabilité empirique
def proba_empirique(d):
    total = len(d)
    if total == 0:
        return {0: 0.0, 1: 0.0}  # Cas particulier ma liste vide
    
    # Compte les éléments de chaque classe
    count = {0: 0, 1: 0}
    for point in d:
        count[point.y] += 1
    
    # Calcule les proportions
    proba = {0: count[0] / total, 1: count[1] / total}
    return proba


# Vérification de la condition sur l'attribut
def question_inf(x, a, s):

    return x.x[a] < s

# Séparation des données selon une question
def split(d, question):
    a, s = question
    d1 = []
    d2 = []
    for i in range(len(d)):
        
        if question_inf(d[i], a, s):
            d1.append(d[i])
        else:
            d2.append(d[i])
    return d1, d2

#Liste des seuils possibles pour un attribut
def list_separ_attributs(d, a):
    val_attribut= []
    for i in range(len(d)):
        val_attribut.append(d[i].x[a])

    new_list = []
    new_list = list(set(val_attribut))
    new_list.sort()

    return_list = []
    for i in range(len(new_list)-1):
        return_list.append((a,((new_list[i] + new_list[i+1]) / 2)))

    return return_list

# Liste de toutes les questions possibles pour l'ensemble des attributs
def liste_questions(d):
    cles = list(d[0].x.keys()) 
    questions = []
    for a in cles:
        questions.extend(list_separ_attributs(d,a))
    return questions
    
# Calcul de l'entropie 
def entropie(d):
    pi = proba_empirique(d)
    
    return -(pi[0]* math.log2(pi[0]) + pi[1] * math.log2(pi[1]))

# Calcule de l'entropie 
def entropie(d):
    pe = proba_empirique(d)
    
    entropy = 0
    for value in pe.values():
        if value > 0:
            entropy -= value * math.log2(value)
    
    return entropy


# Calcule du gain d'entropie
def gain_entropie(d, question):
    d1, d2 = split(d, question)
    r1 = len(d1) / len(d)
    r2 = len(d2) / len(d)
    
    return entropie(d) - r1*entropie(d1) - r2*entropie(d2)

def best_split(d):
    questions = liste_questions(d)
    max_entropy_gain = 0
    for question in questions:
        if max_entropy_gain < gain_entropie(d, question):
            max_entropy_gain = gain_entropie(d, question)
            best_question = question
        
    return best_question
    
   
