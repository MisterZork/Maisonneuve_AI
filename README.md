# Maisonneuve AI

## Table des matières

- [Description](#description)
- [Installation](#installation)
- [Utilisation](#utilisation)

## Description

Maisonneuve AI est un projet éducatif développé au Collège de Maisonneuve, par l'équipe d'ESP (2025) de Hamza G., Zacky C. et William L.D.
Ce projet implémente un réseau de neurones pour la reconnaissance d'images. Le système est capable de reconnaître différents chiffres à partir du jeu de données EMNIST (Extended MNIST).
Le code inclut aussi 2 parties du dataset de EMNIST (Digits et MNIST), pour simplifier les essais.

## Installation

### Prérequis
- Python 3.8 ou version ultérieure
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. Clonez ce dépôt ou téléchargez l'archive ZIP depuis la section [Releases](https://github.com/MisterZork/Maisonneuve_AI/releases)

```bash
git clone https://github.com/MisterZork/Maisonneuve_AI.git
cd Maisonneuve_AI
```

2. Installez les dépendances requises

```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application, exécutez le script principal depuis le répertoire du projet :

```bash
cd src
python main.py
```

Le programme va être prêt à collecter les datasets dans src/data/datasets et va être capable de s'entraîner à partir d'eux, donnant lieu à un réseau entraîné et sauvegardé dans data/layers/__nom-attribué__.
