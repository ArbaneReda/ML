import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Désactiver les logs inutiles de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Désactiver l'exécution eager pour compatibilité TF1
tf.compat.v1.disable_eager_execution()

# Charger l'environnement FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=True)

# ✅ Réseau de neurones pour approximer Q-values
tf.compat.v1.reset_default_graph()
inputs1 = tf.compat.v1.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random.uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# ✅ Fonction de perte et mise à jour
nextQ = tf.compat.v1.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# ✅ Paramètres d'apprentissage
y = 0.99
e = 1.0
e_decay = 0.995
e_min = 0.01
num_episodes = 2000

# ✅ Variables globales
jList = []
rList = []
stop_training = False

# ✅ Création de l'interface ttkbootstrap
app = ttk.Window(themename="superhero")
app.title("Q-Network Training - FrozenLake")
app.geometry("1000x800")

# ✅ Titre
title_label = ttk.Label(app, text="🏆 Entraînement du Q-Network", font=("Helvetica", 18), bootstyle=PRIMARY)
title_label.pack(pady=10)

# ✅ Barre de progression
progress_var = ttk.IntVar()
progress_bar = ttk.Progressbar(app, length=600, mode="determinate", variable=progress_var, maximum=num_episodes)
progress_bar.pack(pady=10)

# ✅ Label d'épisode
episode_label = ttk.Label(app, text="Épisode: 0/2000", font=("Helvetica", 12))
episode_label.pack(pady=5)

# ✅ Graphiques
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

# ✅ Fonction d'arrêt
def stop():
    global stop_training
    stop_training = True

stop_button = ttk.Button(app, text="⛔ Stop Training", bootstyle=DANGER, command=stop)
stop_button.pack(pady=10)

# ✅ Fonction pour quitter proprement
def quit_app():
    stop()
    app.quit()

quit_button = ttk.Button(app, text="❌ Quitter", bootstyle=SECONDARY, command=quit_app)
quit_button.pack(pady=5)

# ✅ Fonction pour télécharger les graphes
def download_graph():
    fig.savefig("training_results.png")
    print("📥 Graphique enregistré sous 'training_results.png'")

download_button = ttk.Button(app, text="📥 Télécharger Graphique", bootstyle=INFO, command=download_graph)
download_button.pack(pady=10)

# ✅ Entraînement du Q-Network
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

            s = env.reset()[0]
            rAll = 0
            d = False
            j = 0

            while j < 99:
                j += 1
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()

                s1, r, d, _, _ = env.step(a[0])
                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ.copy()
                targetQ[0, a[0]] = r + y * maxQ1

                sess.run(updateModel, feed_dict={inputs1: np.identity(16)[s:s+1], nextQ: targetQ})
                rAll += r
                s = s1

                if d:
                    e = max(e_min, e * e_decay)
                    break

            rList.append(rAll)
            jList.append(j)

            # ✅ Mise à jour des graphes
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
