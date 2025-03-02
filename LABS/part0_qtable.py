import gym
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialisation de l'environnement FrozenLake
env = gym.make("FrozenLake-v1")

# Initialisation de la table Q pour le Q-learning
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Paramètres du Q-learning
lr = .8  # Taux d'apprentissage (learning rate)
y = .95  # Facteur d'actualisation (discount factor)
num_episodes = 2000  # Nombre total d'épisodes
max_steps = 99  # Nombre maximum d'itérations par épisode

# Création de l'interface graphique avec ttkbootstrap
app = ttk.Window(themename="superhero")
app.title("Q-Learning - FrozenLake")
app.geometry("1000x800")

# Ajout des éléments de l'interface utilisateur
title_label = ttk.Label(app, text="🔵 Apprentissage du robot...", font=("Helvetica", 16), bootstyle=PRIMARY)
title_label.pack(pady=10)

progress_var = ttk.IntVar()
progress_bar = ttk.Progressbar(app, length=600, mode="determinate", variable=progress_var, maximum=num_episodes)
progress_bar.pack(pady=10)

# Affichage de l'état de l'entraînement
episode_label_var = ttk.StringVar()
episode_label = ttk.Label(app, textvariable=episode_label_var, font=("Helvetica", 12))
episode_label.pack()

robot_state_var = ttk.StringVar()
robot_state_label = ttk.Label(app, textvariable=robot_state_var, font=("Courier", 12), justify="left")
robot_state_label.pack(pady=10)

# Aperçu de la table Q
qtable_label = ttk.Label(app, text="📋 Q-Table (Aperçu)", font=("Helvetica", 12, "bold"))
qtable_label.pack(pady=5)
qtable_text = ttk.Text(app, height=10, width=60, font=("Courier", 10))
qtable_text.pack()

# Configuration du graphique de suivi des récompenses
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_title("Évolution des récompenses")
ax.set_xlabel("Épisodes")
ax.set_ylabel("Récompenses")
line, = ax.plot([], [], 'r-')

canvas = FigureCanvasTkAgg(fig, master=app)
canvas.get_tk_widget().pack()

# Historique des récompenses
reward_history = []

# Variable d'arrêt de l'entraînement
stop_flag = False

def stop():
    """Définit le drapeau d'arrêt de l'entraînement."""
    global stop_flag
    stop_flag = True

# Bouton pour arrêter l'entraînement
quit_button = ttk.Button(app, text="⛔ Arrêter", bootstyle=DANGER, command=stop)
quit_button.pack(pady=10)

# Fonction principale d'entraînement du Q-Network
def train():
    global stop_flag
    r_list = []

    for i in range(num_episodes):
        if stop_flag:
            break

        s = env.reset()[0]  # Réinitialisation de l'environnement
        rAll = 0  # Récompense cumulée
        d = False  # Indicateur de fin d'épisode
        j = 0  # Nombre d'étapes

        while j < max_steps:
            j += 1
            # Sélection de l'action avec un peu d'exploration aléatoire
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            s1, r, d, _, _ = env.step(a)
            # Mise à jour de la table Q selon l'équation de Q-learning
            Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
            rAll += r
            s = s1

            if d:
                break

        r_list.append(rAll)

        # Mise à jour des indicateurs graphiques
        progress_var.set(i + 1)
        episode_label_var.set(f"📈 Épisode {i+1} / {num_episodes} - Réussite: {np.mean(r_list[-100:]) * 100:.2f}%")
        robot_state_var.set(env.render())

        qtable_text.delete("1.0", "end")
        qtable_text.insert("end", np.array_str(Q[:5]))

        reward_history.append(np.mean(r_list[-100:]))
        line.set_xdata(range(len(reward_history)))
        line.set_ydata(reward_history)
        ax.relim()
        ax.autoscale_view()
        canvas.draw()

        app.update_idletasks()
        time.sleep(0.01)

    # Fin de l'entraînement
    print("\n✅ Entraînement terminé !")
    print("Score moyen:", sum(r_list) / num_episodes)
    print("Q-Table finale:\n", Q)

    app.quit()

# Démarrage de l'entraînement dans un thread séparé
def start():
    global stop_flag
    stop_flag = False
    training_thread = threading.Thread(target=train)
    training_thread.start()

# Bouton pour démarrer l'entraînement
start_button = ttk.Button(app, text="▶️ Démarrer", bootstyle=SUCCESS, command=start)
start_button.pack(pady=10)

# Lancement de l'interface graphique
app.mainloop()
