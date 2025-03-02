from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from collections import deque
import cv2

# Désactivation du mode eager pour compatibilité avec TensorFlow 1.x
tf.compat.v1.disable_eager_execution()

# Paramètres d'entraînement améliorés
batch_size = 32
y = 0.99  
startE = 1  
endE = 0.1  
annealing_steps = 10000.0  
num_episodes = 10000  
pre_train_steps = 10000  
max_epLength = 50  
h_size = 512  
update_freq = 4  
tau = 0.001  
path = "./dqn_model"  

if not os.path.exists(path):
    os.makedirs(path)

# Chargement de l’environnement gridworld
from gridworld import gameEnv

env = gameEnv(partial=False, size=5)
action_size = env.actions

# Classe Experience Replay
class ExperienceBuffer:
    def __init__(self, buffer_size=100000):  
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, size):
        return random.sample(self.buffer, size)

# Normalisation des états
def processState(state):
    frame_resized = cv2.resize(state, (84, 84))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    return frame_normalized  # Garder en 3D

# Mise à jour progressive du réseau cible
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = [tfVars[idx + total_vars // 2].assign(
        tfVars[idx] * tau + tfVars[idx + total_vars // 2] * (1 - tau)
    ) for idx in range(total_vars // 2)]
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

# Réseau de neurones convolutif (Dueling DQN)
class QNetwork:
    def __init__(self, h_size, action_size):
        self.scalarInput = tf.compat.v1.placeholder(shape=[None, 84, 84, 3], dtype=tf.float32)

        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(self.scalarInput)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(conv2)
        conv4 = tf.keras.layers.Conv2D(filters=h_size, kernel_size=7, strides=1, activation='relu')(conv3)

        streamAC, streamVC = tf.split(conv4, num_or_size_splits=2, axis=3)
        streamA = tf.keras.layers.Flatten()(streamAC)
        streamV = tf.keras.layers.Flatten()(streamVC)

        xavier_init = tf.keras.initializers.GlorotUniform()
        self.AW = tf.Variable(xavier_init(shape=(h_size // 2, action_size)))
        self.VW = tf.Variable(xavier_init(shape=(h_size // 2, 1)))

        self.Advantage = tf.matmul(streamA, self.AW)
        self.Value = tf.matmul(streamV, self.VW)

        # Correction du calcul de Qout
        self.Qout = self.Value + (self.Advantage - tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, axis=1)

        # Perte et optimisation
        self.targetQ = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

mainQN = QNetwork(h_size, action_size)
targetQN = QNetwork(h_size, action_size)

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()
trainables = tf.compat.v1.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = ExperienceBuffer()

# Politique d'exploration améliorée
epsilon = startE
total_steps = 0
rewards_list = []

# Fonction d'affichage en temps réel avec suppression après affichage
def plot_rewards(rewards):
    plt.figure(figsize=(12,6))
    plt.plot(rewards, label="Reward per episode", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.show(block=False)
    plt.pause(2)  # Laisse le graphique affiché 2 secondes avant suppression
    plt.close()  # Ferme la figure pour éviter d'en accumuler trop

with tf.compat.v1.Session() as sess:
    sess.run(init)
    updateTarget(targetOps, sess)
    
    for episode in range(num_episodes):
        state = processState(env.reset())
        episode_reward = 0

        for step in range(max_epLength):
            if np.random.rand() < epsilon or total_steps < pre_train_steps:
                action = np.random.randint(0, action_size)
            else:
                action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.expand_dims(state, axis=0)})[0]

            next_state, reward, done = env.step(action)
            next_state = processState(next_state)
            myBuffer.add((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps > pre_train_steps and total_steps % update_freq == 0:
                batch = myBuffer.sample(batch_size)
                states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = zip(*batch)

                Q1 = sess.run(mainQN.Qout, feed_dict={mainQN.scalarInput: np.array(next_states_mb)})
                Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.array(next_states_mb)})

                # Correction définitive pour Double Q-Learning
                best_action_indexes = np.argmax(Q1, axis=1)
                doubleQ = np.take_along_axis(Q2, np.expand_dims(best_action_indexes, axis=1), axis=1).squeeze()

                target_Qs = rewards_mb + y * doubleQ * (1 - np.array(dones_mb, dtype=float))
                sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.array(states_mb), mainQN.targetQ: target_Qs, mainQN.actions: actions_mb})
                updateTarget(targetOps, sess)

            if done:
                break

        print(f"▶️ Épisode {episode} terminé - Reward total: {episode_reward}")

        rewards_list.append(episode_reward)
        
        if episode % 100 == 0:
            plot_rewards(rewards_list)  # Mise à jour du graphique

        # Politique d'exploration améliorée (progressive)
        if epsilon > endE:
            epsilon = max(endE, epsilon * 0.999)

    saver.save(sess, path + "/final_model.ckpt")
    plt.plot(np.convolve(rewards_list, np.ones(10)/10, mode='valid'))
    plt.show()
