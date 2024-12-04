import numpy as np
import math


class DataPoint:
    # x c'est les attributs
    # y c'est les classes
    # cles c'est le nom des attributs
    def __init__(self, x, y, cles):
        self.x = {}  # dic
        for i in range(len(cles)):
            self.x[cles[i]] = float(x[i])
        self.y = int(y)
        self.dim = len(self.x)

    def __repr__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y)


def load_data(filelocation):
    with open(filelocation, "r") as f:
        data = []
        attributs = f.readline()[:-1].split(",")[:-1]
        for line in f:
            z = line.split(",")
            if z[-1] == "\n":
                z = z[:-1]
            x = z[:-1]
            y = int(z[-1])
            data.append(DataPoint(x, y, attributs))
    return data


class Noeud:
    def __init__(self, profondeur_max=np.infty):
        self.question = None
        self.enfants = {}
        self.profondeur_max = profondeur_max
        self.proba = None
        self.hauteur = None

    def prediction(self, x):
        # Ici on regarde si on est sur une feuille on va retourner la proba
        if self.proba is not None:
            return self.proba

        # Ici si on est pas sur une feuille on va regarder appliquer la question recurssivement
        if self.question is not None:
            question_attribut, question_seuil = self.question
            if x[question_attribut] <= question_seuil:
                return self.enfants["enfant_1"].prediction(x)
            else:
                return self.enfants["enfant_2"].prediction(x)

    def grow(self, data, depth=0):
        entropy = entropie(data)
        question = best_split(data)

        if question is None:
            self.hauteur = depth
            self.proba = proba_empirique(data)
            return

        d1, d2 = split(data, question)

        if entropy > 0 and depth < self.profondeur_max and len(d1) > 0 and len(d2) > 0:
            self.question = question
            self.enfants["enfant_1"] = Noeud(self.profondeur_max)
            self.enfants["enfant_2"] = Noeud(self.profondeur_max)

            self.enfants["enfant_1"].hauteur = depth + 1
            self.enfants["enfant_2"].hauteur = depth + 1

            self.enfants["enfant_1"].grow(d1, depth + 1)
            self.enfants["enfant_2"].grow(d2, depth + 1)
        else:
            self.hauteur = depth
            self.proba = proba_empirique(data)

    def calculate_node_cost(self, alpha, data=None):
        if data is None:
            if self.proba is not None:
                return alpha
            
            return float('inf')
        
        # Calculate misclassification
        misclassification = 0
        for point in data:
            prediction = self.prediction(point.x)
            pred_value = max(prediction.values())
            for key, value in prediction.items():
                if pred_value == value:
                    classe_predite = key
                    break
            
            if point.y != classe_predite:
                misclassification += 1
        
        # Cost is misclassification error plus complexity penalty
        return misclassification + alpha

    def elagage(self, alpha, data=None):
        # If this is a leaf node, return its cost
        if self.proba is not None:
            return self.calculate_node_cost(alpha, data)
        
        # If not a leaf, first prune children
        if "enfant_1" in self.enfants and "enfant_2" in self.enfants:
            cost_1 = self.enfants["enfant_1"].elagage(alpha, data)
            cost_2 = self.enfants["enfant_2"].elagage(alpha, data)
        
        # Calculate the cost of pruning this node (turning it into a leaf)
        leaf_cost = self.calculate_node_cost(alpha, data)
        
        # Calculate the current subtree cost
        if "enfant_1" in self.enfants and "enfant_2" in self.enfants:
            subtree_cost = cost_1 + cost_2
        else:
            subtree_cost = float('inf')
        
        # Compare leaf cost with subtree cost
        if leaf_cost <= subtree_cost:
            # Prune the subtree by converting to a leaf
            self.enfants.clear()  # Remove children
            self.question = None  # Remove splitting question
            self.proba = proba_empirique([])  # Set probability based on current data
            return leaf_cost
        
        return subtree_cost


# Calcul de la probabilité empirique
def proba_empirique(d):
    total = len(d)

    if total == 0:
        return {}

    classes = {}
    for i in range(total):
        if d[i].y not in classes:
            classes[d[i].y] = 0
        classes[d[i].y] += 1

    return {cles: values / total for cles, values in classes.items()}


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


# Liste des seuils possibles pour un attribut
def list_separ_attributs(d, a):
    val_attribut = []
    for i in range(len(d)):
        val_attribut.append(d[i].x[a])

    val_attribut = list(set(val_attribut))
    val_attribut.sort()

    return_list = []
    for i in range(len(val_attribut) - 1):
        return_list.append((a, ((val_attribut[i] + val_attribut[i + 1]) / 2)))

    return return_list


# Liste de toutes les questions possibles pour l'ensemble des attributs
def liste_questions(d):
    questions = []
    for a in d[0].x.keys():
        questions.extend(list_separ_attributs(d, a))
    return questions


# Calcul de l'entropie
def entropie(d):
    pe = proba_empirique(d)
    entropy = 0
    for proba in pe.values():
        if proba > 0:
            entropy += proba * math.log2(proba)
    return -entropy


# Calcule du gain d'entropie
def gain_entropie(d, question):
    d1, d2 = split(d, question)
    if len(d1) == 0 or len(d2) == 0:
        return 0

    r1 = len(d1) / len(d)
    r2 = len(d2) / len(d)

    return entropie(d) - r1 * entropie(d1) - r2 * entropie(d2)


def best_split(d):
    best_question = None
    questions = liste_questions(d)
    max_entropy_gain = -float("inf")

    for question in questions:
        if max_entropy_gain < gain_entropie(d, question):
            max_entropy_gain = gain_entropie(d, question)
            best_question = question

    return best_question


def precision(node, data):
    count = 0
    for d in data:
        prediction = node.prediction(d.x)
        pred_value = max(prediction.values())
        for key, value in prediction.items():
            if pred_value == value:
                classe_predite = key
        if d.y == classe_predite:
            count += 1

    return count / len(data) * 100 if data else 0


def dataSet_Split(data):
    copy = data.copy()
    np.random.shuffle(copy)
    train_set_size = int(len(copy) * 0.8)
    train_set = copy[:train_set_size]
    test_set = copy[train_set_size:]
    return train_set, test_set