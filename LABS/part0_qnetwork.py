import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Désactivation des logs inutiles de TensorFlow pour améliorer la lisibilité des sorties
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Désactivation de l'exécution eager pour garantir la compatibilité avec TensorFlow 1
tf.compat.v1.disable_eager_execution()

# Chargement de l'environnement FrozenLake (avec une surface glissante pour complexifier l'apprentissage)
env = gym.make("FrozenLake-v1", is_slippery=True)

# Définition du réseau de neurones pour approximer les Q-values
tf.compat.v1.reset_default_graph()
inputs1 = tf.compat.v1.placeholder(shape=[1, 16], dtype=tf.float32)  # Entrée du réseau : état actuel
W = tf.Variable(tf.random.uniform([16, 4], 0, 0.01))  # Poids du réseau
Qout = tf.matmul(inputs1, W)  # Calcul des valeurs Q
predict = tf.argmax(Qout, 1)  # Action prédite (celle ayant la plus grande valeur Q)

# Définition de la fonction de perte et de la mise à jour des poids
nextQ = tf.compat.v1.placeholder(shape=[1, 4], dtype=tf.float32)  # Valeurs cibles des Q-values
loss = tf.reduce_sum(tf.square(nextQ - Qout))  # Erreur quadratique
trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)  # Optimiseur de descente de gradient
updateModel = trainer.minimize(loss)  # Mise à jour du modèle

# Paramètres d'apprentissage
y = 0.99  # Facteur de discount
num_episodes = 2000  # Nombre total d'épisodes
e = 1.0  # Taux d'exploration initial
e_decay = 0.995  # Facteur de décroissance de l'exploration
e_min = 0.01  # Taux minimal d'exploration

# Variables globales pour stocker les statistiques
jList = []  # Nombre d'étapes par épisode
rList = []  # Récompenses par épisode
stop_training = False  # Variable pour arrêter l'entraînement

# Création de l'interface graphique avec ttkbootstrap
app = ttk.Window(themename="superhero")
app.title("Q-Network Training - FrozenLake")
app.geometry("1000x800")

# Ajout des widgets de l'interface utilisateur
title_label = ttk.Label(app, text="🏆 Entraînement du Q-Network", font=("Helvetica", 18), bootstyle=PRIMARY)
title_label.pack(pady=10)

progress_var = ttk.IntVar()
progress_bar = ttk.Progressbar(app, length=600, mode="determinate", variable=progress_var, maximum=num_episodes)
progress_bar.pack(pady=10)

episode_label = ttk.Label(app, text="Épisode: 0/2000", font=("Helvetica", 12))
episode_label.pack(pady=5)

# Configuration des graphiques pour visualiser la progression de l'entraînement
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
fig.subplots_adjust(hspace=0.5)

ax1.set_title("Success Rate Over Time")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Success Rate")

ax2.set_title("Steps per Episode Over Time")
ax2.set_xlabel("Episodes")
ax2.set_ylabel("Steps")

line1, = ax1.plot([], [], 'b-', label="Success Rate")
line2, = ax2.plot([], [], 'r-', label="Steps per Episode")

ax1.legend()
ax2.legend()

canvas = FigureCanvasTkAgg(fig, master=app)
canvas.get_tk_widget().pack()

# Fonction pour arrêter l'entraînement
def stop():
    global stop_training
    stop_training = True

stop_button = ttk.Button(app, text="⛔ Stop Training", bootstyle=DANGER, command=stop)
stop_button.pack(pady=10)

# Fonction pour quitter proprement l'application
def quit_app():
    stop()
    app.quit()

quit_button = ttk.Button(app, text="❌ Quitter", bootstyle=SECONDARY, command=quit_app)
quit_button.pack(pady=5)

# Fonction pour sauvegarder les graphiques
def download_graph():
    fig.savefig("training_results.png")
    print("📥 Graphique enregistré sous 'training_results.png'")

download_button = ttk.Button(app, text="📥 Télécharger Graphique", bootstyle=INFO, command=download_graph)
download_button.pack(pady=10)

# Fonction principale d'entraînement du Q-Network
def train():
    global e
    stop_training = False
    init = tf.compat.v1.global_variables_initializer()
    
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            if stop_training:
                break

            episode_label.config(text=f"📈 Épisode: {i+1}/{num_episodes}")
            progress_var.set(i+1)
            app.update_idletasks()

            s = env.reset()[0]  # Réinitialisation de l'environnement
            rAll = 0  # Récompense cumulée
            d = False  # Indicateur de fin d'épisode
            j = 0  # Nombre d'étapes

            while j < 99:
                j += 1
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()  # Exploration

                s1, r, d, _, _ = env.step(a[0])
                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ.copy()
                targetQ[0, a[0]] = r + y * maxQ1  # Mise à jour des Q-values

                sess.run(updateModel, feed_dict={inputs1: np.identity(16)[s:s+1], nextQ: targetQ})
                rAll += r
                s = s1

                if d:
                    e = max(e_min, e * e_decay)  # Réduction de l'exploration
                    break

            rList.append(rAll)
            jList.append(j)

            # Mise à jour des graphiques en temps réel
            if len(rList) > 1:
                line1.set_xdata(range(len(rList)))
                line1.set_ydata(rList)
                ax1.relim()
                ax1.autoscale_view()

            if len(jList) > 1:
                line2.set_xdata(range(len(jList)))
                line2.set_ydata(jList)
                ax2.relim()
                ax2.autoscale_view()

            canvas.draw()

            if i % 100 == 0:
                print(f"Épisode {i} - Succès moyen: {sum(rList[-100:]) / 100:.2f}%")

    success_rate = sum(rList) / num_episodes * 100
    print(f"✅ Percent of successful episodes: {success_rate:.2f}%")

start_button = ttk.Button(app, text="▶️ Démarrer", bootstyle=SUCCESS, command=train)
start_button.pack(pady=10)

app.mainloop()
