import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time

# ✅ Désactiver les logs inutiles de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.disable_eager_execution()

# ✅ Charger l'environnement FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=True)

# ✅ Initialisation du modèle
tf.compat.v1.reset_default_graph()
graph = tf.compat.v1.get_default_graph()

with graph.as_default():
    inputs1 = tf.compat.v1.placeholder(shape=[1, 16], dtype=tf.float32)
    W = tf.Variable(tf.random.uniform([16, 4], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout, 1)

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
log_file = "logs.txt"

# ✅ Fonction pour enregistrer les logs
def log(message):
    print(message)  # ✅ Affiche dans la console aussi
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ✅ Fonction d'entraînement
def train():
    global e

    # ✅ Nettoyer les logs précédents
    if os.path.exists(log_file):
        os.remove(log_file)

    with graph.as_default():
        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            for i in range(num_episodes):
                s = env.reset()[0]
                rAll = 0
                d = False
                j = 0

                while j < 99:
                    j += 1
                    a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]})
                    if np.random.rand(1) < e:
                        a[0] = env.action_space.sample()

                    s1, r, terminated, truncated, _ = env.step(a[0])
                    d = terminated or truncated

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

                log(f"📈 Épisode {i+1}/{num_episodes} - Taux de réussite: {sum(rList[-100:]) / 100:.2f}%")

        success_rate = sum(rList) / num_episodes * 100
        log(f"✅ Pourcentage d'épisodes réussis: {success_rate:.2f}%")  # ✅ Gardé en dernier log

        save_results()

# ✅ Sauvegarde les graphes dans un dossier statique
def save_results():
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rList, label="Taux de réussite")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompenses")
    plt.title("Évolution du taux de réussite")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(jList, label="Nombre d'actions par épisode", color='red')
    plt.xlabel("Épisodes")
    plt.ylabel("Nombre d'actions")
    plt.title("Évolution du nombre d'actions par épisode")
    plt.legend()

    # ✅ Sauvegarde dans `backend/static/`
    os.makedirs("static", exist_ok=True)
    file_path = "static/resultsqnetwork.png"
    plt.savefig(file_path)
    
    log(f"📊 Résultats sauvegardés sous `{file_path}`")  # ✅ Affichage correct

# ✅ Démarrer l'entraînement
if __name__ == "__main__":
    train()
