from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from collections import deque
import cv2  # 📷 Pour le traitement d'image

# ✅ Désactivation du mode eager pour compatibilité avec les placeholders de TF1
tf.compat.v1.disable_eager_execution()

# ✅ Paramètres d'entraînement améliorés
batch_size = 32
y = 0.99  
startE = 1  
endE = 0.1  # Exploration plus longue
annealing_steps = 10000.0  # Ajusté pour un meilleur apprentissage
num_episodes = 10000  # Conformité au lab
pre_train_steps = 10000  # Plus de pré-entraînement
max_epLength = 50  
h_size = 512  # Conformité au lab
update_freq = 4  # Plus stable
tau = 0.001  # Meilleure stabilité
path = "./dqn_model"  

if not os.path.exists(path):
    os.makedirs(path)

# ✅ Chargement de l’environnement gridworld
from gridworld import gameEnv

env = gameEnv(partial=False, size=5)
action_size = env.actions

# ✅ Classe Experience Replay
class ExperienceBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, size):
        return random.sample(self.buffer, size)

# ✅ Prétraitement de l'état
def processState(state):
    frame_resized = cv2.resize(state, (84, 84))
    frame_normalized = frame_resized / 255.0  # Normalisation
    return frame_normalized.flatten()

# ✅ Mise à jour du réseau cible
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[:total_vars // 2]):
        target_var = tfVars[idx + total_vars // 2]
        op_holder.append(target_var.assign(tf.multiply(var, tau) + tf.multiply(target_var, (1 - tau))))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

# ✅ Réseau de neurones convolutif (Dueling DQN)
class QNetwork:
    def __init__(self, h_size, action_size):
        self.scalarInput = tf.compat.v1.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        
        # Couches convolutionnelles
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu')(self.imageIn)
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')(self.conv1)
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')(self.conv2)
        self.conv4 = tf.keras.layers.Conv2D(h_size, kernel_size=7, strides=1, activation='relu')(self.conv3)
        
        # Dueling DQN
        self.streamAC, self.streamVC = tf.split(self.conv4, num_or_size_splits=2, axis=3)
        self.streamA = tf.keras.layers.Flatten()(self.streamAC)
        self.streamV = tf.keras.layers.Flatten()(self.streamVC)

        xavier_init = tf.keras.initializers.GlorotUniform()
        self.AW = tf.Variable(xavier_init(shape=(h_size // 2, action_size)))
        self.VW = tf.Variable(xavier_init(shape=(h_size // 2, 1)))

        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
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

# ✅ Entraînement
epsilon = startE
epsilon_decay = (startE - endE) / annealing_steps
total_steps = 0
rewards_list = []

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
                action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [state]})[0]
            
            next_state, reward, done = env.step(action)
            next_state = processState(next_state)
            myBuffer.add((state, action, reward, next_state, done))
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if total_steps > pre_train_steps and total_steps % update_freq == 0:
                batch = myBuffer.sample(batch_size)
                states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = zip(*batch)
                Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(next_states_mb)})
                Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(next_states_mb)})
                doubleQ = Q2[np.arange(batch_size), Q1]
                target_Qs = rewards_mb + y * doubleQ * (1 - np.array(dones_mb, dtype=float))
                sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(states_mb), mainQN.targetQ: target_Qs, mainQN.actions: actions_mb})
                updateTarget(targetOps, sess)

            if done:
                break

        rewards_list.append(episode_reward)
        if epsilon > endE:
            epsilon -= epsilon_decay
    
    saver.save(sess, path + "/final_model.ckpt")

plt.plot(np.convolve(rewards_list, np.ones(10)/10, mode='valid'))
plt.show()
