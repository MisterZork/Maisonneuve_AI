import numpy as np
import os
import time
import matplotlib.pyplot as plt
from neurone import Neurone, perte_func

#=============== Collecte des données ===============#

def collecte_datasets():
    """Collecte et prépare tous les datasets disponibles."""
    datasets_main_path = "data/datasets"

    # Vérification du dossier principal
    try:
        datasets_dir = os.listdir(datasets_main_path)
        if "saved" not in datasets_dir:
            os.makedirs(f"{datasets_main_path}/saved", exist_ok=True)
    except FileNotFoundError:
        print("Aucun dossier trouvé dans le répertoire data/datasets")
        print("Veuillez rajouter un dossier contenant les datasets")
        return None

    # Collection des datasets
    saved_datasets = []
    new_datasets = []

    # Recherche des fichiers .csv et .npy
    for directory in datasets_dir:
        dir_path = f"{datasets_main_path}/{directory}"
        if os.path.isdir(dir_path):
            datasets = os.listdir(dir_path)
            if directory == "saved":
                for dataset in datasets:
                    if dataset.endswith((".csv", ".npy")):
                        saved_datasets.append(os.path.join(dir_path, dataset))
            else:
                for dataset in datasets:
                    if dataset.endswith((".csv", ".npy")):
                        new_datasets.append(os.path.join(dir_path, dataset))

    # Conversion des .csv en .npy
    for dataset in new_datasets:
        if dataset.endswith(".csv"):
            base_name = os.path.basename(dataset).replace(".csv", ".npy")
            save_path = os.path.join(datasets_main_path, "saved", base_name)
            if not os.path.exists(save_path):
                print(f"Conversion de {dataset} en .npy")
                save_data = np.loadtxt(dataset, delimiter=",")
                np.save(save_path, save_data)
    return saved_datasets


def selection_dataset(datasets):
    """Permet à l'utilisateur de sélectionner un dataset."""
    if not datasets:
        return None

    print("#" * 40)
    print("Voici les datasets disponibles:")
    for i, dataset in enumerate(datasets):
        print(f"{i + 1}. {os.path.basename(dataset).removesuffix('.npy')}")

    dataset_choice = input("Entrez le numéro du dataset que vous voulez utiliser\n-> ")
    while not dataset_choice.isdigit() or int(dataset_choice) not in range(1, len(datasets) + 1):
        dataset_choice = input("Entrez un chiffre valide\n-> ")

    data = np.load(datasets[int(dataset_choice) - 1])
    print(f"Le dataset {os.path.basename(datasets[int(dataset_choice) - 1]).removesuffix('.npy')} a été chargé")

    return data

#=============== Logique du réseau ===============#

def entrainement(neurone, donnees, epochs, batch_size, learning_rate):
    """Entraîne le réseau de neurones et affiche les métriques."""
    n = len(donnees)
    total_batches = n // batch_size

    # Métriques à suivre
    epoch_acc_list = []
    epoch_losses_list = []
    batch_acc_list = []
    batch_losses_list = []

    for i in range(epochs):
        epoch_true, epoch_total, epoch_loss = 0, 0, 0

        # Mélange des données
        np.random.shuffle(donnees)

        # Traitement par mini-batch
        for j in range(0, n, batch_size):
            batch = donnees[j:j + batch_size]
            batch_loss, batch_total, batch_true = 0, 0, 0

            # Traitement de chaque exemple
            for exemple in batch:
                neurone.propagation(exemple)
                neurone.descente_gradient()
                pred_ans = np.argmax(neurone.activations[neurone.nb_couches - 1])
                real_ans = np.argmax(neurone.ans)

                # Suivi des métriques
                batch_total += 1
                if pred_ans == real_ans:
                    batch_true += 1

                # Calcul de la perte
                batch_loss += perte_func(neurone.activations[neurone.nb_couches - 1], neurone.ans)

            # Mise à jour des poids
            neurone.retropropagation()

            # Mise à jour des statistiques
            epoch_true += batch_true
            epoch_total += batch_total
            epoch_loss += batch_loss

            # Enregistrement des métriques par batch
            if batch_total > 0:
                batch_acc_list.append((batch_true / batch_total) * 100)
                batch_losses_list.append(batch_loss / batch_total)

        # Calcul des métriques d'époque
        acc = (epoch_true / epoch_total) * 100 if epoch_total > 0 else 0
        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0

        # Enregistrement des métriques
        epoch_acc_list.append(acc)
        epoch_losses_list.append(avg_loss)

        # Affichage
        print(f"Epoch {i + 1}/{epochs} terminée :")
        print(f"  Perte: {avg_loss:.4f}")
        print(f"  Précision: {acc:.2f}%")

    # Affichage du graphique
    plt.figure(figsize=(7, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    line1, = ax1.plot(range(1, epochs + 1), epoch_acc_list, 'b-', label='Précision (%)')
    ax1.set_xlabel("Nombre d'époque")
    ax1.set_ylabel('Précision (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    line2, = ax2.plot(range(1, epochs + 1), epoch_losses_list, 'r-', label='Perte')
    ax2.set_ylabel('Perte', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    plt.title('Évolution de la précision et de la perte pendant l\'entraînement')
    plt.legend(lines, labels, loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return acc, avg_loss


def test(neurone, donnees, batch_size):
    """Test le réseau de neurones sur des données (sans visualisation)."""
    n = len(donnees)
    total_batches = n // batch_size

    print("#" * 40)
    print(f"Test sur {n} images avec {total_batches} batches")

    total_prediction = 0
    correct_predictions = 0
    neurone.charge_couches()

    for j in range(0, n, batch_size):
        if j % (batch_size * 10) == 0:
            print(f"  Progression: {j // batch_size}/{total_batches} batches")

        mini_batch = donnees[j:min(j + batch_size, n)]

        for exemple in mini_batch:
            neurone.propagation(exemple)

            total_prediction += 1
            pred_ans = np.argmax(neurone.activations[neurone.nb_couches - 1])
            real_ans = np.argmax(neurone.ans)

            if pred_ans == real_ans:
                correct_predictions += 1

    precision = (correct_predictions / total_prediction) * 100 if total_prediction > 0 else 0
    print(f"Test terminé avec une précision de {precision:.2f}%")

    return precision

def showcase(neurone, donnees, batch_size, nb_examples=10):
    """Visualise les prédictions du réseau sur quelques exemples."""
    n = min(len(donnees), nb_examples * batch_size)

    print("#" * 40)
    print(f"Visualisation de {min(n, nb_examples)} exemples")

    neurone.charge_couches()

    # Prendre un échantillon aléatoire
    indices = np.random.choice(len(donnees), n, replace=False)
    examples = donnees[indices]

    for i, exemple in enumerate(examples):
        if i >= nb_examples:
            break

        plot = exemple[1:].reshape(28, 28).T
        neurone.propagation(exemple)

        pred_ans = np.argmax(neurone.activations[neurone.nb_couches - 1])
        real_ans = np.argmax(neurone.ans)

        # Création de la figure pour l'affichage
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Affichage de l'image
        ax1.imshow(plot, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f"Image du chiffre {real_ans}")
        ax1.axis('off')

        # Affichage des probabilités
        probas = np.array(neurone.activations[neurone.nb_couches - 1]).ravel()
        if len(probas) != 10:
            print(f"Attention: Nombre de probabilités incorrect ({len(probas)})")
            probas = np.zeros(10)
            probas[pred_ans] = 1.0  # Fallback en cas d'erreur

        ax2.bar(range(10), probas)
        ax2.set_xlabel('Chiffre')
        ax2.set_ylabel('Probabilité')
        ax2.set_title(f"Prédiction: {pred_ans} {'✓' if pred_ans == real_ans else '✗'}")
        ax2.set_xticks(range(10))

        plt.tight_layout()
        plt.show()

        if i < nb_examples - 1:
            choix = input("Continuer (Entrée) ou Quitter (q) ? ")
            if choix.lower() == 'q':
                break

def sauvegarder_couches(neurone, parametres, accuracy=None, loss=None):
    """Sauvegarde les poids et biais du réseau de neurones."""
    timestamp = int(time.time())

    print("#" * 40)
    print(f"Tentative de sauvegarde des poids et biais à {time.strftime('%H:%M:%S', time.localtime())}")

    # Création des dossiers
    parent_folder = f'./data/layers/{timestamp}_test/'
    os.makedirs(parent_folder, exist_ok=True)

    poids_folder = f'{parent_folder}/poids/'
    biais_folder = f'{parent_folder}/biais/'
    os.makedirs(poids_folder, exist_ok=True)
    os.makedirs(biais_folder, exist_ok=True)

    # Création du fichier d'info
    with open(f"{parent_folder}/info.txt", "w") as file:
        file.write(f"Taille des couches de neurones : {neurone.nb_neurone}\n")
        file.write(f"Timestamp (en secondes) : {timestamp}\n")
        file.write(f"Loss de l'entraînement: {loss if loss is not None else 'UNKNOWN'}\n")
        file.write(f"Précision: {accuracy if accuracy is not None else 'UNKNOWN'}\n")
        file.write(f"Nombre d'époques : {parametres.get('epoch', 'UNKNOWN')}\n")
        file.write(f"Taille des mini-batch : {parametres.get('batch_size', 'UNKNOWN')}\n")
        file.write(f"Taux d'apprentissage : {parametres.get('learning_rate', 'UNKNOWN')}\n")
        file.write(f"Fonction d'activation utilisé : {parametres.get('activation_function', 'UNKNOWN')}\n")

    # Sauvegarde des poids et biais
    for num in range(1, neurone.nb_couches):
        np.save(f"{poids_folder}/poids-{num}.npy", neurone.poids[num])
        np.save(f"{biais_folder}/biais-{num}.npy", neurone.biais[num])

    print("Poids et biais sauvegardés avec succès!")

#=============== Programme principal ===============#

def main():
    """Fonction principale."""
    # Paramètres d'entraînement
    parametres = {
        "epoch": 5,
        "batch_size": 2,
        "learning_rate": 0.8,
        "activation_function": 0  # 0: Sigmoid, 1: ReLU, 2: ELU
    }

    # Collecte des datasets
    datasets = collecte_datasets()
    if not datasets:
        quit("Il n'y a pas de datasets")

    # Sélection d'un dataset
    data = selection_dataset(datasets)
    if data is None:
        quit("Collecte de données impossible")

    # Initialisation du réseau de neurones
    reseau = Neurone([784, 16, 16, 10], parametres) #CHNAGER LA TAILLE DU RÉSEAU ICI

    # Choix du mode d'utilisation
    mode = int(input("Que souhaitez-vous faire avec le réseau ?\n"
                     "Entraîner : 0\nTester : 1\nVisualiser : 2\n-> "))

    if mode == 0:  # Entraînement
        accuracy, loss = entrainement(
            reseau,
            data,
            parametres["epoch"],
            parametres["batch_size"],
            parametres["learning_rate"]
        )
        sauvegarder_couches(reseau, parametres, accuracy, loss)
        return None
    elif mode == 1:  # Test (statistiques)
        test(reseau, data, parametres["batch_size"])
        return None
    elif mode == 2:  # Showcase (visualisation)
        showcase(reseau, data, parametres["batch_size"])
        return None
    else:
        quit("Choix invalide")

if __name__ == "__main__":
    main()