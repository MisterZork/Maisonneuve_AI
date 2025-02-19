import numpy as np

#==================== Fonction de mathématique ====================#
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def fonction_de_cout(y_predit, y_reel):
    return (y_predit - y_reel) ** 2

#Détail important: il faut utiliser ReLu avant softmax (dernière couche)fin d'éviter les problèmes!!!
def softmax(x): #Fonction qui sert à transformer les reps d'un output en pourcentage. (La somme des valeurs=1). ex) [2.3, 1.1, 0.5] devient [0.7, 0.2, 0.1] signifiant qu'il y a 70% de chance que ce soit la première classe, 20% la 2e, etc.
    exp_x = np.exp(x - np.max(x))  # Soustraction pour stabilité numérique
    return exp_x / np.sum(exp_x)

#==================== Classe du réseau neuronal ====================#
class Neurone:
    def __init__(self, *nb_neurone, learning_rate=1):
        self.layer_sizes = nb_neurone[0]
        self.num_layers = len(self.layer_sizes)
        self.learning_rate = learning_rate
        self.couches, self.biais, self.poids, self.z = {}, {}, {}, {}

        for i in range(self.num_layers):
            couche_nom = f'couche_{i}'
            self.couches[couche_nom] = np.zeros(self.layer_sizes[i])
            if i < self.num_layers - 1:
                self.poids[f'w_{i}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.75
                self.biais[f'b_{i}'] = np.full((self.layer_sizes[i + 1]), 1.0)
                self.z[f'z_{i}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))

    def avancement(self):
        for i in range(self.num_layers - 1):
            self.z[f'z_{i}'] = np.dot(self.couches[f'couche_{i}'], self.poids[f'w_{i}']) + self.biais[f'b_{i}']
            self.couches[f'couche_{i + 1}'] = softmax(relu(self.z[f'z_{i}']))

    def backpropagation(self, y_reel):
        y_predit = self.couches[f'couche_{self.num_layers - 1}']
        for i in range(self.num_layers - 2, -1, -1):
            self.poids[f'w_{i}'] -= self.learning_rate * self.derive_partiel_w(y_predit, y_reel, i)
            self.biais[f'b_{i}'] -= self.learning_rate * self.derive_partiel_b(y_predit, y_reel, i)

    def derive_partiel_b(self, y_predit, y_reel, iteration):
        return 2 * np.sum(y_predit - y_reel) * relu_prime(self.z[f'z_{iteration}'])

    def derive_partiel_w(self, y_predit, y_reel, iteration):
        delta = 2 * np.sum(y_predit - y_reel) * relu_prime(self.z[f'z_{iteration}'])
        layer_prev = self.couches[f'couche_{iteration}'].reshape(-1, 1)  # reshape as column vector
        return np.dot(layer_prev, delta.reshape(1, -1))  # outer product

    def set(self, new_img):
        self.couches = {}
        for i in range(self.num_layers):
            couche_nom = f'couche_{i}'
            self.couches[couche_nom] = np.zeros(self.layer_sizes[i])
        self.couches['couche_0'] = new_img

    def caching(self, io, nom):  # Ajouter le paramètre de pouvoir choisir le nom du fichier
        if io == 'save':
            data = self.poids, self.biais
            np.save(f'{nom}.npy', data)
        elif io == 'load':
            data = np.load(f'{nom}.npy')

if __name__ == "__main__":
    print("Test du programme - Essai")
    print(relu(np.array([-2, 1, 42, -16])))
    for i in range(4 - 1, -1, -1):
        print(i)