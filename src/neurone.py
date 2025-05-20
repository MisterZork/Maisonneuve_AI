import numpy as np
import os

#Ce code est là oû se trouve la majorité des mathématiques reliée à l'IA, ainsi que le réseau en lui-même.
#==================== Classe IA ====================#
class Neurone:
    def __init__(self, nb_neurone:list, parametres:dict):
        #On récupère les informations sur les couches du réseau
        self.nb_neurone = nb_neurone
        self.nb_couches = len(self.nb_neurone)

        #On collecte les paramètres qui seront utilisés
        self.parametres = parametres
        self.batch_size = self.parametres['batch_size']
        self.epoch = self.parametres['epoch']
        self.learning_rate = self.parametres['learning_rate']
        self.mode = self.parametres['activation_function']

        #On crée des dictionnaires contenant toutes les couches qu'on utilisera...
        self.activations, self.poids, self.biais, self.z = {}, {}, {}, {}
        self.cache_poids, self.cache_biais, self.delta = {}, {}, {} #Ainsi que des dictionnaires pour le gradient

        #Remplissage des dictionnaires par les couches, poids et biais intéressant
        for l in range(self.nb_couches):
            if l == 0:
                self.activations[l] = np.zeros((self.nb_neurone[l], 1))
                continue
            self.poids[l] = np.random.normal(0, 0.5, (self.nb_neurone[l], self.nb_neurone[l-1])) #Aléatoire
            self.cache_poids[l] = np.zeros((self.nb_neurone[l], self.nb_neurone[l-1]))
            self.biais[l] = np.random.normal(0, 0.25, (self.nb_neurone[l], 1)) #Aléatoire
            self.cache_biais[l] = np.zeros((self.nb_neurone[l], 1))
            self.z[l] = np.zeros((self.nb_neurone[l], 1))

        #On aura besoin de plus tard l'information sur la réponse attendue de l'IA
        self.ans = np.zeros((10, 1))

    #=============== Propagation et rétropropagation ===============#
    def propagation(self, img_data:np.ndarray):
        """
        Cette fonction effectue, étape par étape, l'avancement de la prédiction du réseau de neurone, couche par couche.
        Il récupère les données de l'image (img_data : ndarray) et, selon son mode d'activation, effectue les calculs
        nécessaires pour l'activation des neurones suivants. Note : La prédiction est faite one-on-one (si la prédiction
        est vrai ou faux, sur 100).
        """
        self.charge_image(img_data) # Obtient la première couche
        for l in range(1, self.nb_couches):
            self.z[l] = np.add(np.dot(self.poids[l], self.activations[l - 1]), self.biais[l])
            if self.mode == 0:
                self.activations[l] = sigmoid(self.z[l])
            elif self.mode == 1:
                self.activations[l] = relu(self.z[l])
            else:
                raise Exception("Mode d'activation invalide")

    def descente_gradient(self):
        """
        Cette fonction effectue le calcul du gradient une seule fois pour chaque poids et biais, prêt à être utilisé
        lorsqu'elle est demandé. Le calcul matriciel est basé sur le livre de Micheal Nielsen.
        """

        self.delta = {} # Reset des valeurs
        if np.all(self.activations[self.nb_couches - 1] == 0):
            raise Exception("Aucune prédiction n'a été faite")
        if self.mode != 0 and self.mode != 1:
            raise Exception("Mode d'activation invalide")

        # Calcul du delta de la couche de sortie et des autres couches
        couche_final = self.nb_couches - 1
        if self.mode == 0:  # Sigmoid
            self.delta[couche_final] = np.multiply((self.activations[couche_final] - self.ans), sigmoid_prime(self.z[couche_final]).reshape(-1, 1))
            for l in range(couche_final - 1, 0, -1):
                self.delta[l] = np.multiply(np.dot(self.poids[l + 1].T, self.delta[l + 1]),
                                            sigmoid_prime(self.z[l]).reshape(-1, 1))
        else:  # ReLU
            self.delta[couche_final] = np.multiply((self.activations[couche_final] - self.ans), relu_prime(self.z[couche_final]).reshape(-1, 1))
            for l in range(couche_final - 1, 0, -1):
                self.delta[l] = np.multiply(np.dot(self.poids[l + 1].T, self.delta[l + 1]),
                                            relu_prime(self.z[l]).reshape(-1, 1))

        # Calcul des gradients pour les poids et les biais
        for l in range(1, self.nb_couches):
            self.cache_poids[l] += np.dot(self.delta[l], self.activations[l - 1].T)
            self.cache_biais[l] += self.delta[l]

    def retropropagation(self):
        """
        Cette fonction met à jour les poids et les biais du réseau de neurones, à partir des changements amenés à la fonction
        de la descente du gradient.
        """
        self.descente_gradient()
        for l in range(1, self.nb_couches):
            self.poids[l] -= (self.learning_rate * self.cache_poids[l]) / self.batch_size
            self.biais[l] -= (self.learning_rate * self.cache_biais[l]) / self.batch_size
            self.cache_poids[l] = np.zeros((self.nb_neurone[l], self.nb_neurone[l - 1]))
            self.cache_biais[l] = np.zeros((self.nb_neurone[l], 1))

    #=============== Sauvegarde et chargement ===============#
    def charge_image(self, img_data):
        """
        Cette fonction charge les données de l'image, séparant les pixels de la réponse attendue.
        :param img_data:
        :return:
        """
        self.activations[0] = np.matrix(img_data[1::]).reshape(-1, 1) / 255.0
        self.ans = np.zeros((10, 1))
        self.ans[int(img_data[0])] = 1

    def charge_couches(self):
        """
        Cette fonction charge les poids et les biais du réseau de neurones, à partir d'un fichier .npy.
        :return:
        """
        #TODO : Check if the path is really absolute and won't be perturbed by some modification
        #TODO : Check if the directory/file exists, else recreate the dir and raise Exception
        #On essaye de trouver tous les timestamps des fichiers de poids et de biais
        parent_folder = 'data/layers/'
        trainings_folders = os.listdir(parent_folder)

        #On vérifie s'il existe des fichiers de poids et de biais compatibles
        info_list = []
        for file in trainings_folders:
            with open(f"{parent_folder}{file}/info.txt", 'r') as f:
                if f.readline().endswith(f"Taille des couches de neurones : {self.nb_neurone}\n"):
                    info_list.append(file)

        if len(info_list) == 0:
            raise Exception("Aucun fichier de poids et de biais compatible n'a été trouvé")

        #On demande à l'utilisateur de choisir le timestamp qu'il veut utiliser
        print("Voici les couches disponibles pour ce réseau:")
        for i, dossier in enumerate(info_list):
            print(f"{i + 1}. {os.path.basename(parent_folder)}{dossier}")
        dossier_choix = input("Entrez le numéro des couches que vous voulez utiliser\n-> ")
        while not dossier_choix.isdigit() or int(dossier_choix) not in range(1, len(info_list) + 1):
            dossier_choix = input("Entrez un chiffre valide\n-> ")

        dossier_choisi = info_list[int(dossier_choix) - 1]
        print(f"Les couches de {os.path.basename(info_list[int(dossier_choix) - 1])} a été chargé")

        #TODO : Change the timestamp from an UUID, where the user choose the one to utilize
        #On prend le dernier timestamp
        for i in range(1, self.nb_couches):
            poids_file = os.path.join(parent_folder, dossier_choisi, f"poids/poids-{i}.npy")
            biais_file = os.path.join(parent_folder, dossier_choisi, f"biais/biais-{i}.npy")
            self.poids[i] = np.load(poids_file)
            self.biais[i] = np.load(biais_file)

#==================== Mathematics functions ====================#
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)


def perte_func(y_pred, y_ans):
    return 1/len(y_pred) * np.sum(np.power((y_pred - y_ans), 2))

def perte_prime(y_pred, y_ans):
    return 2/len(y_pred) * np.sum(y_pred - y_ans)