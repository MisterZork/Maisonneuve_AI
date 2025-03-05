# C'est la base du réseau de neurone. Il contient les fonctions d'activation, les fonctions de coût, les fonctions de
# dérivées, etc. Il est utilisé par le neurone pour effectuer les calculs nécessaires à l'entraînement et à la prédiction.

#==================== Importation des modules ====================#
import numpy as np
import os

#==================== Classe du neurone ====================#
class Neurone:
    def __init__(self, nb_neurone:list, mode=0):
        #On récupère les informations sur les couches du réseau
        self.nb_couches = len(nb_neurone)
        self.nb_neurone = nb_neurone

        #On crée des dictionnaires contenant toutes les couches qu'on utilisera
        self.activations, self.poids, self.biais, self.z = {}, {}, {}, {}
        self.cache_poids, self.cache_biais, self.delta = {}, {}, {} #Ainsi que des dictionnaires pour le gradient

        for l in range(self.nb_couches):
            if l == 0:
                self.activations[l] = np.zeros((self.nb_neurone[l], 1))
                continue
            self.poids[l] = np.random.normal(0, 2, (nb_neurone[l], nb_neurone[l-1]))
            self.cache_poids[l] = np.random.normal(0, 2, (nb_neurone[l], nb_neurone[l-1]))
            self.biais[l] = np.random.normal(0, 2, (nb_neurone[l], 1))
            self.cache_biais[l] = np.random.normal(0, 2, (nb_neurone[l], 1))
            self.z[l] = np.random.normal(0, 2, (nb_neurone[l], 1))

        #On aura besoin de plus tard l'information sur la réponse attendue de l'IA
        self.ans = np.zeros((10, 1))
        self.mode = mode # 0 = Sigmoïde; 1 = ReLU; 2 = ELU (Il y en a d'autres à venir)

    def load_image(self, img_data):
        """
        Cette fonction charge les données de l'image, séparant les pixels de la réponse attendue.
        :param img_data:
        :return:
        """
        self.activations[0] = np.matrix(img_data[1::]).reshape(-1, 1) / 255.0
        self.ans = np.zeros((10, 1))
        self.ans[int(img_data[0])] = 1

    def load_couches(self):
        """
        Cette fonction charge les poids et les biais du réseau de neurones, à partir d'un fichier .npy.
        :param timestamp:
        :return:
        """
        #On essaye de trouver tous les timestamps des fichiers de poids et de biais
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data", "layers")
        timestamps = []
        for file in os.listdir(data_dir):
            if os.path.basename(file).endswith(".npy"):
                timestamps.append(os.path.basename(file)[8:-4])

        #On prend le dernier timestamp
        timestamp = sorted(timestamps)[-1]
        for i in range(1, self.nb_couches):
            poids_file = os.path.join(data_dir, f"poids-{i}-{timestamp}.npy")
            biais_file = os.path.join(data_dir, f"biais-{i}-{timestamp}.npy")
            self.poids[i] = np.load(poids_file)
            self.biais[i] = np.load(biais_file)

    def avancement(self, img_data):
        """
        Cette fonction effectue, étape par étape, l'avancement de la prédiction du réseau de neurone, couche par couche.
        Il récupère les données de l'image (img_data : ndarray) et, selon son mode d'activation, effectue les calculs
        nécessaires pour l'activation des neurones suivants.
        :param img_data:
        :return:
        """
        self.load_image(img_data)
        for l in range(1, self.nb_couches):
            self.z[l] = np.add(np.dot(self.poids[l], self.activations[l - 1]), self.biais[l])
            if self.mode == 0:
                self.activations[l] = sigmoid(self.z[l])
            elif self.mode == 1:
                self.activations[l] = relu(self.z[l])
            elif self.mode == 2:
                self.activations[l] = elu(self.z[l])
            else:
                raise Exception("Mode d'activation invalide")

    def gradient(self):
        """
        Cette fonction effectue le calcul du gradient une seule fois pour chaque poids et biais, prêt à être utilisé
        lorsqu'elle est demandé.
        :return:
        """
        if np.all(self.activations[self.nb_couches - 1] == 0):
            raise Exception("Aucune prédiction n'a été faite")
        self.delta = {} # Reset des valeurs

        # Calcul du delta de la couche de sortie
        couche_final = self.nb_couches - 1
        if self.mode == 0:  # Sigmoid
            self.delta[couche_final] = np.multiply((self.activations[couche_final] - self.ans), sigmoid_prime(self.z[couche_final]).reshape(-1, 1))
        elif self.mode == 1:  # ReLU
            self.delta[couche_final] = np.multiply((self.activations[couche_final] - self.ans), relu_prime(self.z[couche_final]).reshape(-1, 1))

        # Calcul des deltas des couches cachées
        for l in range(couche_final - 1, 0, -1):
            if self.mode == 0:  # Sigmoid
                self.delta[l] = np.multiply(np.dot(self.poids[l + 1].T, self.delta[l + 1]), sigmoid_prime(self.z[l]).reshape(-1, 1))
            elif self.mode == 1:  # ReLU
                self.delta[l] = np.multiply(np.dot(self.poids[l + 1].T, self.delta[l + 1]), relu_prime(self.z[l]).reshape(-1, 1))

        # Calcul des gradients pour les poids et les biais
        for l in range(1, self.nb_couches):
            self.cache_poids[l] = np.dot(self.delta[l], self.activations[l - 1].T)
            self.cache_biais[l] = self.delta[l]

    def descente_gradient_normal(self, learning_rate):
        """
        Cette fonction met à jour les poids et les biais du réseau de neurones en utilisant la descente de gradient
        normale.
        :param learning_rate:
        :return:
        """
        self.gradient()
        for l in range(1, self.nb_couches):
            self.poids[l] -= learning_rate * self.cache_poids[l]
            self.biais[l] -= learning_rate * self.cache_biais[l]

    def descente_gradient_stochastic(self, learning_rate, mini_batch_size, data):
        """
        Cette fonction effectue en bouche la descente de gradient stochastique, en utilisant des mini-batches
        pour entraîner le réseau de neurones.
        :param learning_rate:
        :param mini_batch_size:
        :param data:
        :return:
        """
        for i in range(mini_batch_size):
            self.avancement(data[i])
            self.gradient()
        for l in range(1, self.nb_couches):
            self.poids[l] -= (self.cache_poids[l] * learning_rate)/ mini_batch_size
            self.biais[l] -= (self.cache_biais[l] * learning_rate)/ mini_batch_size

#==================== Fonctions mathématiques ====================#
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def elu(x):
    return np.where(x > 0, x, 0.01 * (np.exp(x) - 1))

def elu_prime(x):
    return np.where(x > 0, 1, 0.01 * np.exp(x))

def perte_func(y_pred, y_ans):
    return 1/len(y_pred) * np.sum((y_pred - y_ans) ** 2)

def perte_prime(y_pred, y_ans):
    return 2/len(y_pred) * np.sum(y_pred - y_ans)