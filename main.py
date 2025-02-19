import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.textTools import caselessSort

from neurone import Neurone

#==================== Mode d'utilisation ====================#
print("#" * 40)
mode = input(f"Sélectionnez le mode d'utilisation\n"
             f"{"-" * 15}\n1. Entraînement\n2. Test\n"
             f"{"-" * 15}\n-> ")

while True:
    if mode in ["1", "2"]:
        break
    else:
        mode = input("Veuillez entrer un chiffre valide\n-> ")

#==================== Collecte de l'image du dataset ====================#
print("#" * 40)
dataset = input(f"Sélectionnez le dataset\n"
                 f"{"-" * 15}\n1. Balanced (Experimental)\n2. Class (Experimental)\n3. Merge (Experimental)\n4. Digits\n5. Letters (Experimental)\n6. MNIST\n"
                 f"{'-' * 15}\n-> ")
while True:
    if dataset in ["1", "2", "3", "4", "5", "6"]:
        break
    else:
        dataset = input("Veuillez entrer un chiffre valide\n-> ")

#==================== Réseau de neurones====================#
good_guess, bad_guess, total_guess = 0, 0, 0
dict_dataset = {"1": "balanced", "2": "class", "3": "merge", "4": "digits", "5": "letters", "6": "mnist"}
dict_results = {"1": 47, "2": 62, "3": 47, "4": 10, "5": 26, "6": 10}
dict_total_number = {"1": 131600, "2": 814255, "3": 814255, "4": 280000, "5": 103600, "6": 70000}
neurone = Neurone([784, 26, 18, dict_results[dataset]], 2.5)

print("#" * 40)

#==================== Entraînement ====================#
for i in range(50000):
    img_id = i
    img = np.loadtxt(f"Datasets/EMNIST/emnist-{dict_dataset[dataset]}-{"train" if mode == "1" else "test"}.csv",
                     delimiter=",", max_rows=1, skiprows=(img_id))
    img_val = int(img[0])
    rep_np = np.zeros(dict_results[dataset])
    rep_np[img_val] = 1
    neurone.set(img[1::])
    neurone.avancement()
    neurone.backpropagation(rep_np)
    total_guess += 1
    if img_val == np.argmax(neurone.couches['couche_3']):
        good_guess += 1
    else:
        bad_guess += 1
    print(f"Prédiction: {np.argmax(neurone.couches['couche_3'])} | Réel: {img_val} | "
          f"{"✔️" if img_val == np.argmax(neurone.couches['couche_3']) else "☠️"} | Itération: {i} | "
          f"Pourcentage: {round((good_guess / total_guess) * 100, 2)}%")

#==================== Affichage du résultat ====================#
print(neurone.couches["couche_3"])


