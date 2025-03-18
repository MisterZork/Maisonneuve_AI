# C'est le programme principal du projet AI. Son but principal est de charger n'importe quel type de données d'images du
# répertoire /Datasets et d'entraîner le réseau de neurones pour reconnaître les images. L'utilisateur peut choisir entre
# l'entraînement et le test du réseau.

#==================== Importation des modules ====================#
from neurone import *
from telemetry import *
import os
import time
import matplotlib.pyplot as plt

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
epoch = 1
batch_size = 1
n = len(data) # Pour le moment, il va utiliser 100% du dataset; à changer plus tard, si on effectue + d'époques
total_batches = n // batch_size
learning_rate = 0.8
activation_function = 0 # 0: Sigmoid, 1: ReLU, 2: ELU

#TODO : ASK THE USER IF THESE VARIABLES ARE OKAY
#On va revérifier avec la console pour voir si c'est volontaire ou non

if batch_size * total_batches < n:
    print("Le nombre d'images n'est pas divisible par la taille du mini-batch. Certaines images ne seront pas utilisées")
#==================== Réseau de neurones ====================#
# Le réseau de neurones est initialisé avec les paramètres suivants :
neurone = Neurone([784, 16, 16, 10], activation_function)
acc = 0

#==================== Entraînement / Test ====================#
# Le code va demander à l'utilisateur s'il veut créer un entraînement ou un test du réseau de neurones.
#TODO : Ajouter le showcase
mode_utilisation = int(input("Voulez-vous entraîner ou tester le réseau de neurone ?\n"
                         "Entraîner : 0\nTester : 1\n-> "))

# La fonction de descente de gradient stochastique est utilisée pour entraîner le réseau de neurones.
def training(e=epoch, bs=batch_size, lr=learning_rate, d=data):
    test_true, test_total, test_loss = 0, 0, 0
    batch_dict = {}
    global acc
    for i in range(e):
        print(f"Epoch {i + 1}/{e} started")
        epoch_true, epoch_total, epoch_loss = 0, 0, 0

        # Shuffle data at the beginning of each epoch
        np.random.shuffle(d)

        # Create mini-batches and do the SGD
        for j in range(0, n, bs):
            batch_dict[j] = d[j:j + bs]
            batch_loss, batch_total, batch_true = 0, 0, 0

            #Prédiction du réseau de chaque image
            for k in range(bs):
                neurone.avancement(batch_dict[j][k])
                neurone.gradient()
                pred_ans = np.argmax(neurone.activations[neurone.nb_couches - 1])
                real_ans = np.argmax(neurone.ans)

                # Track metrics
                batch_total += 1
                if pred_ans == real_ans:
                    batch_true += 1


            # Update weights and biases
            neurone.update(lr, bs)

            # Calculate loss
            # batch_loss += perte_func(neurone.activations[neurone.nb_couches - 1], neurone.ans)

            # Update epoch statistics
            epoch_true += batch_true
            epoch_total += bs
            epoch_loss += batch_loss

            # Print batch progress periodically
            if (j // bs) % 5 == 0:
                print(f"  Batch {j // bs}/{total_batches}: "
                      f"Loss: {batch_loss / bs:.4f}, "
                      f"Accuracy: {(batch_true / bs) * 100:.2f}%")

        # Calculate epoch metrics
        test_total += epoch_total
        test_true += epoch_true
        acc = (test_true / test_total) * 100
        avg_loss = epoch_loss / test_total

        # Print epoch results
        print(f"Epoch {i + 1}/{e} finished with:")
        # print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {acc:.2f}%")

def testing():
    #TODO : CHANGER LE CODE ENTIER DE TESTING
    print("#" * 40)
    print(f"Test sur {n} images avec {total_batches} batches")
    total_prediction = 0
    correct_predictions = 0
    neurone.load_couches()
    for j in range(0, n, batch_size):
        if j % (batch_size * 10) == 0:
            print(f"  Progress: {j // batch_size}/{total_batches} batches")

        # Get the mini-batch (ensure we don't go out of bounds)
        end_idx = min(j + batch_size, n)
        mini_batch = data[j:end_idx]


        # Process each example in the mini-batch
        for k in range(len(mini_batch)):
            plot = mini_batch[k][1::].reshape(28, 28).T
            neurone.avancement(mini_batch[k])
            total_prediction += 1
            pred_ans = np.argmax(neurone.activations[neurone.nb_couches - 1])
            real_ans = np.argmax(neurone.ans)
            if pred_ans == real_ans:
                correct_predictions += 1

            print(f"Prediction : {pred_ans} / Réponse : {real_ans} / Precision : {(correct_predictions / total_prediction) * 100:.4f} %"
                  f" / {"✔" if pred_ans == real_ans else "✘"}")
            plt.imshow(plot, cmap='gray', vmin=0, vmax=255)
            plt.show()
            time.sleep(1)

    print(f"Test finished with an accuracy of {(correct_predictions / total_prediction) * 100:.2f} %")

def showcase():
    #TODO : CRÉER LE CODE DE SHOWCASE POUR MONTRER LES IMAGES PAR TEST
    pass


def save_couches(accuracy=acc):
    """
    Cette fonction sauvegarde les poids et les biais du réseau de neurones dans un fichier .npy.
    :return:
    """
    timestamp = (time.time() // 1)

    print("#" * 40)
    print(f"Tentative de sauvegarde des poids et biais à {time.strftime('%H:%M:%S', time.localtime())}")

    #Crée un dossier pour l'entraînement
    parent_folder = f'./data/layers/{int(timestamp)}_test/'
    os.mkdir(parent_folder)

    #Crée un sous-dossier pour les poids et les biais
    poids_folder = f'{parent_folder}/poids/'
    biais_folder = f'{parent_folder}/biais/'
    os.mkdir(poids_folder)
    os.mkdir(biais_folder)

    #Création d'un fichier texte pour stocker les informations sur le réseau
    with open(f"{parent_folder}/info.txt", "w") as file:
        file.write(f"Taille des couches de neurones : {neurone.nb_neurone}\n")
        file.write(f"Timestamp (en secondes) : {timestamp}\n")
        file.write(f"Loss de l'entraînement: UNKNOWN\n") #TODO : ADD THE LOSS
        file.write(f"Nombre d'époques : {epoch}\n")
        file.write(f"Taille des mini-batch : {batch_size}\n")
        file.write(f"Taux d'apprentissage : {learning_rate}\n")
        file.write(f"Fonction d'activation utilisé : {activation_function}\n")

    # Sauvegarde des poids et biais dans ce dossier
    for num in range(1, neurone.nb_couches):
        np.save(f"{poids_folder}/poids-{num}.npy", neurone.poids[num])
        np.save(f"{biais_folder}/biais-{num}.npy", neurone.biais[num])

    print("Poids et biais sauvegardé :)")

if mode_utilisation == 0:
    training()
    save_couches()
elif mode_utilisation == 1:
    testing()
else:
    print("Choix invalide")
