# 📌 Q-Learning & Deep Q-Network (DQN) sur FrozenLake

Ce projet implémente différentes méthodes d'apprentissage par renforcement sur l'environnement **FrozenLake** de OpenAI Gym, avec une progression allant de la **Q-Table** à des modèles plus avancés comme le **Q-Network** et le **Double DQN**.

## 📂 Contenu du Projet

- `part0_qtable.py` → Implémentation classique de **Q-Learning** avec une **Q-Table**.
- `part0_qnetwork.py` → Implémentation du **Q-Network** (utilisant un réseau de neurones au lieu d'une table).
- `part4_dqn.py` → Implémentation d'un **Double DQN**, une version améliorée du DQN pour éviter certaines erreurs d’apprentissage.
- `gridworld.py` → Code pour générer un environnement personnalisé, utilisé dans le **lab4**.
- `requirements.txt` → Liste des bibliothèques Python nécessaires à l'exécution du projet.

---

## ⚙️ Installation & Pré-requis

### 📥 1. Cloner le projet

```bash
 git clone <lien-du-repo-github>
 cd <nom-du-repo>
```

### 📦 2. Installer les dépendances

Le fichier `requirements.txt` contient toutes les bibliothèques nécessaires.

#### 🖥️ Pour Windows / Mac / Linux
```bash
pip install -r requirements.txt
```

Si vous utilisez **Anaconda**, créez un environnement dédié :
```bash
conda create --name rl_env python=3.9
conda activate rl_env
pip install -r requirements.txt
```

---

## 🚀 Exécution des scripts

### 🔹 **Q-Table (Q-Learning classique)**

Ce script utilise une **table de valeurs Q** pour entraîner un agent à naviguer dans une grille **4x4** de FrozenLake.

```bash
python part0_qtable.py
```

Ce modèle est rapide et converge efficacement, car FrozenLake est un environnement simple.

---

### 🔹 **Q-Network (Réseau de neurones)**

Ce script remplace la **Q-Table** par un **réseau de neurones** qui approxime les valeurs d’action.

```bash
python part0_qnetwork.py
```

📌 **Attention :** Le Q-Network **prend plus de temps** à s'entraîner que la Q-Table car il doit ajuster les poids d’un modèle neuronal.

---

### 🔹 **Lab 4 - Double DQN (Apprentissage avancé)**

Ce script implémente un **Double DQN**, une version améliorée du DQN classique qui réduit le biais d'optimisme des valeurs Q.

```bash
python part4_dqn.py
```

📌 **Important :** 
- L'entraînement est **beaucoup plus long**, car il utilise un réseau de neurones profond et un buffer d’expérience.
- L'environnement utilisé est **gridworld.py**, une grille personnalisée où l’agent doit naviguer en évitant des obstacles.

---

## ⏳ Temps d'exécution estimé

- **Q-Table** → ⚡ **Rapide** (~ quelques secondes/minutes)
- **Q-Network** → ⚡ **Rapide** (~ quelques secondes/minutes)
- **Double DQN (Lab 4)** → 🕒 **Long** (~ plusieurs minutes/heures, selon les paramètres)

---

## 📝 Conclusion

Ce projet illustre comment un agent peut apprendre à **prendre des décisions intelligentes** en utilisant différentes approches d’apprentissage par renforcement. 

La **Q-Table** est efficace pour les petits environnements, tandis que le **Q-Network** et le **Double DQN** permettent d’étendre cette approche à des scénarios plus complexes.

Bon entraînement ! 🚀