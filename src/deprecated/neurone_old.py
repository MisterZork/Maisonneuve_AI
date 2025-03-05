import numpy as np

#==================== Fonction de mathématique ====================#
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def logistic(x):
    return x / (1 + x)

def fonction_de_cout(y_predit, y_reel):
    return (y_predit - y_reel) ** 2

#Détail important: il faut utiliser ReLu avant softmax (dernière couche)fin d'éviter les problèmes!!!
def softmax(x): #Fonction qui sert à transformer les reps d'un output en pourcentage. (La somme des valeurs=1). ex) [2.3, 1.1, 0.5] devient [0.7, 0.2, 0.1] signifiant qu'il y a 70% de chance que ce soit la première classe, 20% la 2e, etc.
    exp_x = np.exp(x - np.max(x))  # Soustraction pour stabilité numérique
    return exp_x / np.sum(exp_x)

#==================== Code de vérification ====================#
def compatibility_test():
    pass

#==================== Classe du réseau neuronal ====================#
class Neurone:
    def __init__(self, *nb_neurone, learning_rate=1):
        self.layer_sizes = nb_neurone[0]
        self.num_layers = len(self.layer_sizes)
        self.learning_rate = learning_rate
        self.activations, self.biais, self.poids, self.z, self.cache_w, self.cache_b, self.delta = {}, {}, {}, {}, {}, {}, {}

        for i in range(self.num_layers):
            couche_nom = f'couche_{i}'
            self.activations[couche_nom] = np.zeros(self.layer_sizes[i])
            if i < self.num_layers - 1:
                self.poids[f'w_{i}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1])
                self.cache_w[f'cache_w_{i}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
                self.biais[f'b_{i}'] = np.random.random_sample(self.layer_sizes[i + 1])
                self.cache_b[f'cache_b_{i}'] = np.zeros(self.layer_sizes[i + 1])
                self.z[f'z_{i}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))

    def avancement(self, image, mode='sigmoid'):
        self.set(image)
        for i in range(self.num_layers - 1):
            self.z[f'z_{i}'] = np.dot(self.activations[f'couche_{i}'], self.poids[f'w_{i}']) + self.biais[f'b_{i}']
            if i != (self.num_layers - 2):
                if mode == 'sigmoid':
                    self.activations[f'couche_{i + 1}'] = sigmoid(self.z[f'z_{i}'])
                elif mode == 'relu':
                    self.activations[f'couche_{i + 1}'] = relu(self.z[f'z_{i}'])
                else:
                    exit("Mode invalide")
            else:
                if mode == 'sigmoid':
                    self.activations[f'couche_{i + 1}'] = softmax(self.z[f'z_{i}'])

    def backpropagation(self, y_reel, iter=0, activation='sigmoid'):
        self.delta = {}

        y_predit = self.activations[f'couche_{self.num_layers - 1}']
        for i in range(self.num_layers - 2, -1, -1):
            if activation == "sigmoid":
                self.delta_sigmoid(y_predit, y_reel, i)
            elif activation == "relu":
                self.delta_relu(y_predit, y_reel, i)

            # Calculate gradients
            self.cache_w[f'cache_w_{i}'] = self.learning_rate * np.dot(self.activations[f'couche_{i}'].reshape(-1, 1), self.delta.T)
            self.cache_b[f'cache_b_{i}'] = self.learning_rate * np.squeeze(np.mean(self.delta, axis=1, keepdims=True))

        if iter == 0:
            for i in range(self.num_layers - 2, -1, -1):
                self.poids[f'w_{i}'] -= self.cache_w[f'cache_w_{i}']
                self.biais[f'b_{i}'] -= self.cache_b[f'cache_b_{i}']

                # Reset
                self.cache_w[f'cache_w_{i}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
                self.cache_b[f'cache_b_{i}'] = np.zeros(self.layer_sizes[i + 1])


    def delta_relu(self, y_predit, y_reel, i):
        #Dernière couche
        if self.num_layers - 2 == i:
            y_predit = np.array(y_predit).flatten()
            y_reel = np.array(y_reel).flatten()
            error = y_predit - y_reel
            self.delta[i] = np.multiply(error, relu_prime(self.z[f'z_{i}']).reshape(-1, 1))
        #Couches cachées
        else:
            delta_av = self.delta[i+1]
            poids_av = self.poids[f'w_{i + 1}']
            self.delta[i] = np.multiply(np.dot(poids_av, delta_av), relu_prime(self.z[f'z_{i}']))

    def delta_sigmoid(self, y_predit, y_reel, i):
        if self.num_layers - 2 == i:
            y_predit = np.array(y_predit).flatten()
            y_reel = np.array(y_reel).flatten()
            error = y_predit - y_reel
            self.delta = np.multiply(error, sigmoid_prime(self.z[f'z_{i}']).reshape(-1, 1))
        else:
            delta_av = self.delta[i + 1]
            poids_av = self.poids[f'w_{i + 1}']
            self.delta[i] = np.multiply(np.dot(poids_av, delta_av), sigmoid_prime(self.z[f'z_{i}']))

    def set(self, new_img):
        self.activations = {}
        for i in range(self.num_layers):
            couche_nom = f'couche_{i}'
            self.activations[couche_nom] = np.zeros(self.layer_sizes[i])
        self.activations['couche_0'] = new_img

    def caching(self, io, nom="interrupted_data"):  # Ajouter le paramètre de pouvoir choisir le nom du fichier
        if io == 'save':
            data = self.poids, self.biais
            np.save(f'{nom}.npy', data)
        elif io == 'load':
            data = np.load(f'{nom}.npy')

#==================== Test du module ====================#
if __name__ == "__main__":
    neurone = Neurone([784, 24, 12, 10], 0.05)
    neurone.set(np.random.random_sample(784))
    neurone.avancement()
    neurone.backpropagation(np.random.random_sample(10), 1)
    neurone.caching('save')
    neurone.caching('load')
    print("Test réussi")