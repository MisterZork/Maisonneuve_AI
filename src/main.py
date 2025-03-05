# C'est le programme principal du projet AI. Son but principal est de charger n'importe quel type de données d'images du
# répertoire /Datasets et d'entraîner le réseau de neurones pour reconnaître les images. L'utilisateur peut choisir entre
# l'entraînement et le test du réseau.
import numpy as np

#==================== Importation des modules ====================#
from neurone import *
import os
import time

#==================== Obtention des datasets ====================#
# Le but de cette section est de récupérer tous les datasets disponibles dans le répertoire /data/Datasets.
datasets_main_path = "data/Datasets"
try:
    datasets_dir = os.listdir(datasets_main_path)
    if "saved" not in datasets_dir:
        os.mkdir(f"{datasets_main_path}/saved")
except FileNotFoundError:
    exit("Aucun dossier se trouve dans le répertoire data/Datasets\n"
         "Veuillez rajouter un dossier contenant les datasets")

# Il y aura en tout temps un dossier "saved" qui contiendra les datasets déjà utilisé par l'utilisateur.
saved_datasets = []
new_datasets = []

# Recherche récursive des fichiers .csv et .npy dans les dossiers du répertoire /Datasets
for directory in datasets_dir:
    datasets = os.listdir(f"{datasets_main_path}/{directory}")
    if directory == "saved":
        for dataset in datasets:
            if dataset.endswith((".csv", ".npy")):
                saved_datasets.append(os.path.join(datasets_main_path, directory, dataset))
    else:
        for dataset in datasets:
            if dataset.endswith((".csv", ".npy")):
                new_datasets.append(os.path.join(datasets_main_path, directory, dataset))

# Il faut que le programme (en parallèle ou non) transforme les fichiers .csv en .npy pour une utilisation plus rapide.
for dataset in new_datasets:
    if dataset.endswith(".csv"):
        base_name = os.path.basename(dataset).replace(".csv", ".npy")
        save_path = os.path.join(datasets_main_path, "saved", base_name)
        if save_path not in saved_datasets:
            print(f"Conversion de {dataset} en .npy")
            save_data = np.loadtxt(os.path.join(datasets_main_path, dataset), delimiter=",")
            np.save(save_path, save_data)

# Le programme doit maintenant demander à l'utilisateur quel dataset il veut accéder
print("#" * 40)
print("Voici les datasets disponibles:")
for i, dataset in enumerate(saved_datasets):
    print(f"{i + 1}. {os.path.basename(dataset).removesuffix(".npy")}")

dataset_choice = input("Entrez le numéro du dataset que vous voulez utiliser\n-> ")
while not dataset_choice.isdigit() or int(dataset_choice) not in range(1, len(saved_datasets) + 1):
    dataset_choice = input("Entrez un chiffre valide\n-> ")

data = np.load(saved_datasets[int(dataset_choice) - 1])
print(f"Le dataset {os.path.basename(saved_datasets[int(dataset_choice) - 1]).removesuffix('.npy')} a été chargé")

#==================== Initialisation des variables ====================#
# Les variables suivantes sont utilisées pour l'entraînement du réseau de neurones.
epoch = 10
mini_batch_size = 300
n = len(data)
total_batches = n // mini_batch_size
learning_rate = 0.025
activation_function = 0 # 0: Sigmoid, 1: ReLU, 2: ELU

#On va revérifier avec la console pour voir si c'est volontaire ou non

#==================== Réseau de neurones ====================#
# Le réseau de neurones est initialisé avec les paramètres suivants :
neurone = Neurone([784, 18, 18, 10], activation_function)

def perte_func(y_pred, y_ans):
    return 1/len(y_pred) * np.sum(np.array(y_pred - y_ans) ** 2)

#==================== Entraînement / Test ====================#
# Le code va demander à l'utilisateur s'il veut créer un entraînement ou un test du réseau de neurones.
mode_utilisation = int(input("Voulez-vous entraîner ou tester le réseau de neurone ?\n"
                         "Entraîner : 0\nTester : 1\n-> "))

# La fonction de descente de gradient stochastique est utilisée pour entraîner le réseau de neurones.
def training():
    print("#" * 40)
    print(f"Entraînement sur {n} images avec {total_batches} batches par epoch")
    for e in range(epoch):
        epoch_loss = 0
        correct_predictions = 0
        total_prediction = 0

        print(f"Epoch {e + 1}/{epoch} started")
        np.random.shuffle(data)

        for j in range(0, n, mini_batch_size):
            if j % (mini_batch_size * 10) == 0:
                print(f"  Progress: {j // mini_batch_size}/{total_batches} batches")

            # Get the mini-batch (ensure we don't go out of bounds)
            end_idx = min(j + mini_batch_size, n)
            mini_batch = data[j:end_idx]
            batch_loss = 0

            # Process each example in the mini-batch
            for k in range(len(mini_batch)):
                # Forward pass
                neurone.avancement(mini_batch[k])

                # Calculate loss
                loss = perte_func(neurone.activations[neurone.nb_couches - 1], neurone.ans)
                batch_loss += loss

                # Update weights after every example (stochastic)
                neurone.descente_gradient_normal(learning_rate)

                # Count correct predictions
                total_prediction += 1
                if np.argmax(neurone.activations[neurone.nb_couches - 1]) == np.argmax(neurone.ans):
                    correct_predictions += 1

            # Update epoch loss
            epoch_loss += batch_loss / len(mini_batch)

        print(f"Epoch {e + 1}/{epoch} finished with a loss of {epoch_loss:.2f} and an accuracy of "
              f"{(correct_predictions / total_prediction) * 100:.2f} %")

def testing():
    print("#" * 40)
    print(f"Test sur {n} images avec {total_batches} batches")
    total_prediction = 0
    correct_predictions = 0
    for j in range(0, n, mini_batch_size):
        if j % (mini_batch_size * 10) == 0:
            print(f"  Progress: {j // mini_batch_size}/{total_batches} batches")

        # Get the mini-batch (ensure we don't go out of bounds)
        end_idx = min(j + mini_batch_size, n)
        mini_batch = data[j:end_idx]

        # Process each example in the mini-batch
        for k in range(len(mini_batch)):
            neurone.load_couches()
            neurone.avancement(mini_batch[k])
            total_prediction += 1
            if np.argmax(neurone.activations[neurone.nb_couches - 1]) == np.argmax(neurone.ans):
                correct_predictions += 1

        print(f"Test finished with an accuracy of {(correct_predictions / total_prediction) * 100:.2f} %")

def save_couches(acc=0):
    """
    Cette fonction sauvegarde les poids et les biais du réseau de neurones dans un fichier .npy.
    :return:
    """
    timestamp = time.time_ns()
    print("#" * 40)
    print(f"Tentative de sauvegarde des poids et biais à {time.ctime()}")
    for i in range(1, neurone.nb_couches):
        np.save(f"data/layers/poids-{i}-{timestamp}.npy", neurone.poids[i])
        np.save(f"data/layers/biais-{i}-{timestamp}.npy", neurone.biais[i])

    print("Poids et biais sauvegardé :)")

if mode_utilisation == 0:
    training()
    save_couches()
elif mode_utilisation == 1:
    testing()
else:
    print("Choix invalide")
