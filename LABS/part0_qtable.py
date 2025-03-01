import gym
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

env = gym.make("FrozenLake-v1")

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = .8 # taux d'apprentissage
y = .95 # taux de réduction des récompenses
num_episodes = 2000
max_steps = 99

app = ttk.Window(themename="superhero")
app.title("Q-Learning - FrozenLake")
app.geometry("1000x800")

title_label = ttk.Label(app, text="🔵 Apprentissage du robot...", font=("Helvetica", 16), bootstyle=PRIMARY)
title_label.pack(pady=10)

progress_var = ttk.IntVar()
progress_bar = ttk.Progressbar(app, length=600, mode="determinate", variable=progress_var, maximum=num_episodes)
progress_bar.pack(pady=10)

episode_label_var = ttk.StringVar()
episode_label = ttk.Label(app, textvariable=episode_label_var, font=("Helvetica", 12))
episode_label.pack()

robot_state_var = ttk.StringVar()
robot_state_label = ttk.Label(app, textvariable=robot_state_var, font=("Courier", 12), justify="left")
robot_state_label.pack(pady=10)

qtable_label = ttk.Label(app, text="📋 Q-Table (Aperçu)", font=("Helvetica", 12, "bold"))
qtable_label.pack(pady=5)
qtable_text = ttk.Text(app, height=10, width=60, font=("Courier", 10))
qtable_text.pack()

fig, ax = plt.subplots(figsize=(5, 3))
ax.set_title("Évolution des récompenses")
ax.set_xlabel("Épisodes")
ax.set_ylabel("Récompenses")
line, = ax.plot([], [], 'r-')

canvas = FigureCanvasTkAgg(fig, master=app)
canvas.get_tk_widget().pack()

reward_history = []

stop_flag = False
def stop():
    global stop_flag
    stop_flag = True

quit_button = ttk.Button(app, text="⛔ Arrêter", bootstyle=DANGER, command=stop)
quit_button.pack(pady=10)

def train():
    global stop_flag
    r_list = []

    for i in range(num_episodes):
        if stop_flag:
            break

        s = env.reset()[0]
        rAll = 0
        d = False
        j = 0

        while j < max_steps:
            j += 1
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            s1, r, d, _, _ = env.step(a)
            Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
            rAll += r
            s = s1

            if d:
                break

        r_list.append(rAll)

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

    print("\n✅ Entraînement terminé !")
    print("Score moyen:", sum(r_list) / num_episodes)
    print("Q-Table finale:\n", Q)

    app.quit()

def start():
    global stop_flag
    stop_flag = False
    training_thread = threading.Thread(target=train)
    training_thread.start()

start_button = ttk.Button(app, text="▶️ Démarrer", bootstyle=SUCCESS, command=start)
start_button.pack(pady=10)

app.mainloop()
