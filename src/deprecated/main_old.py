import numpy as np
from neurone_old import Neurone

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
mode_data = input(f"Sélectionnez le dataset\n"
                 f"{"-" * 15}\n1. Balanced (Experimental)\n2. Class (Experimental)\n3. Merge (Experimental)\n4. Digits\n5. Letters (Experimental)\n6. MNIST\n"
                 f"{'-' * 15}\n-> ")
while True:
    if mode_data in ["1", "2", "3", "4", "5", "6"]:
        break
    else:
        mode_data = input("Veuillez entrer un chiffre valide\n-> ")

#Note : Les datasets sont basé sur celui de EMNIST
dict_dataset = {"1": "balanced", "2": "class", "3": "merge", "4": "digits", "5": "letters", "6": "mnist"}
dict_results = {"1": 47, "2": 62, "3": 47, "4": 10, "5": 26, "6": 10}
dict_train_number = {"1": 112800, "2": 697932, "3": 697932, "4": 240000, "5": 88800, "6": 60000}
dict_test_number = {"1": 18800, "2": 116323, "3": 116323, "4": 40000, "5": 88800, "6": 60000}

#==================== Réseau de neurones====================#
neurone = Neurone([784, 49, 16, dict_results[mode_data]], 0.01)
print("#" * 40)

#==================== Récupération du dataset ====================#
dataset = np.loadtxt(
    f"src/data/Datasets/EMNIST/emnist-{dict_dataset[mode_data]}-{"train" if mode == "1" else "test"}.csv", delimiter=",")
#[[img 1], [img 2], [img 3], ...] #SHUFFLE

#==================== Entraînement ====================#
def stochastic_gradient_descent(data, epoch, mini_batch_size):
    n = len(data)
    total_batches = n // mini_batch_size
    print(f"Entraînement sur {n} images avec {total_batches} batches par epoch")

    for i in range(epoch):
        epoch_loss = 0
        correct_predictions = 0

        print(f"Epoch {i+1}/{epoch} started")
        # Shuffle data at the beginning of each epoch
        np.random.shuffle(data)

        # Create mini-batches
        for j in range(0, n, mini_batch_size):
            if j % (mini_batch_size * 10) == 0:  # Show progress every 10 batches
                print(f"  Progress: {j//mini_batch_size}/{total_batches} batches")

            mini_batch = data[j:j + mini_batch_size]
            batch_loss = 0

            for i in range(neurone.num_layers - 1):
                neurone.cache_w[f'cache_w_{i}'] = np.zeros((neurone.layer_sizes[i], neurone.layer_sizes[i + 1]))
                neurone.cache_b[f'cache_b_{i}'] = np.zeros(neurone.layer_sizes[i + 1])

            # Process each example in the mini-batch
            for k, example in enumerate(mini_batch):
                img_val = int(example[0])
                img_data = example[1:] / 255.0

                # Forward pass
                neurone.avancement(img_data)

                # Create target (one-hot encoding)
                target = np.zeros(neurone.layer_sizes[-1])
                target[img_val] = 1

                # Calculate loss
                output = neurone.activations[f'couche_{neurone.num_layers - 1}']
                batch_loss += np.sum((output - target) ** 2)

                # Track accuracy
                prediction = np.argmax(output)
                if img_val == prediction:
                    correct_predictions += 1

                # Backward pass only when the batch is complete
                is_last = (k == len(mini_batch) - 1)
                neurone.backpropagation(target, 0 if is_last else 1)

            epoch_loss += batch_loss / mini_batch_size

        # Print epoch summary
        accuracy = correct_predictions / n * 100
        print(f"Epoch {i+1}/{epoch} completed - Loss: {epoch_loss/total_batches:.4f} - Accuracy: {accuracy:.2f}%")

if mode == "1":
    good_guess, bad_guess, total_guess = 0, 0, 0
    print("Entraînement démarré")
    stochastic_gradient_descent(dataset, 20, 50)

    # Evaluate on some test samples
    test_samples = dataset[:100]  # Use first 100 samples for evaluation
    for sample in test_samples:
        img_val = int(sample[0])
        img_data = sample[1:] / 255.0

        neurone.avancement(img_data)
        total_guess += 1
        prediction = np.argmax(neurone.activations[f'couche_{neurone.num_layers - 1}'])
        if img_val == prediction:
            good_guess += 1
        else:
            bad_guess += 1

    print(f"Accuracy: {round((good_guess / total_guess) * 100, 2)}%")

#==================== Affichage du résultat ====================#
print(neurone.activations["couche_3"])


