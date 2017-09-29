
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import os
import sys
import random
from collections import deque, namedtuple

np.random.seed(1)
tf.set_random_seed(1)


# In[1]:

class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        st_shape,
        learning_rate=0.01,
        reward_decay=0.9,
        epsilon_start=0.9,
        replace_target_iter=300,
        memory_size=20000,
        batch_size=32,
        epsilon_decrease=True,
        output_graph=False,
    ):
        self.n_actions=n_actions
        self.width=int(st_shape[0])
        self.height=int(st_shape[1])
        # self.grayscale=True if st_shape[2]=='1' else False
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_decrease =  epsilon_decrease
        self.epsilon_end = 0.001
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start if epsilon_decrease else 0
        self.explore = 100000
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter

        # total_learning_step
        self.learn_step_counter = 0

        # initialize zero memory [state, action, reward, next_state, done]
        self.replay_memory = []
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        
        self._build_model()
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        checkpoint = tf.train.get_checkpoint_state("saved_n")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        if not os.path.exists('saved_n'):
            os.makedirs('saved_n')
        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
    
    
    def make_policy(self, state):
        A =np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        q_values = self.predict(self.sess, np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - self.epsilon)
        if self.epsilon > self.epsilon_end:
            self.epsilon -= ((self.epsilon_start-self.epsilon_end)/self.explore)
        return A
    def store_transition(self, state, action, reward, done, n_state):
        if len(self.replay_memory) == self.memory_size:
            self.replay_memory.pop(0)
        self.replay_memory.append(self.Transition(state, action, reward, n_state, done))
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_parms(self.sess)
            self.saver.save(self.sess, 'saved_n/saved', global_step=self.learn_step_counter)
        samples = random.sample(self.replay_memory, self.batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
        
        q_values = self.preditc(self.sess, states_batch)
        best_actions = np.argmax(q_values, axis=1)
        q_target = self.predict_t(self.sess, next_states_batch)
        targets_batch = reward_batch + self.gamma * q_target[np.arange(self.batch_size), best_actions]
        states_batch = np.array(states_batch)
        loss = self.update(self.sess, np.array(states_batch), np.array(action_batch), np.array(targets_batch))
        self.cost_his.append(loss)
        
        self.learn_step_counter += 1
        
    def predict(self, sess, state):
        return sess.run(self.q_eval, { self.x_pl: state })
    
    def predict_t(self, sess, state):
        return sess.run(self.q_next, { self.x_pl_: state })
    
    def _build_model(self):
        def build_layers(X, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                conv1 = tf.contrib.layers.conv2d(
                    X, 32, 5, 1, activation_fn=tf.nn.relu, padding="VALID",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
                
            with tf.variable_scope('l2'):
                conv2 = tf.contrib.layers.conv2d(
                    pool1, 32, 3, 1, activation_fn=tf.nn.relu, padding="VALID",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
                
            with tf.variable_scope('Value'):
                conv3_v = tf.contrib.layers.conv2d(
                    pool2, 64, 2, 1, activation_fn=tf.nn.relu, padding="VALID",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                pool3_v = tf.nn.max_pool(conv3_v, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
                flattened_v = tf.contrib.layers.flatten(pool3_v)
                fc1_v = tf.contrib.layers.fully_connected(flattened_v , 64, variables_collections=c_names)
                self.V = tf.contrib.layers.fully_connected(fc1_v, 1, activation_fn=None)
                
            with tf.variable_scope('Advantage'):
                conv3_a = tf.contrib.layers.conv2d(
                    pool2, 64, 2, 1, activation_fn=tf.nn.relu, padding="VALID",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                pool3_a = tf.nn.max_pool(conv3_a, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
                flattened_a = tf.contrib.layers.flatten(pool3_a)
                fc1_a = tf.contrib.layers.fully_connected(flattened_a , 64, variables_collections=c_names)
                self.A = tf.contrib.layers.fully_connected(fc1_a, self.n_actions, activation_fn=None)
                
            with tf.variable_scope('Q'):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)) # Q = V(s) + A(s,a)
                
            return out
        self.x_pl = tf.placeholder(tf.float32, [None, 80, 80,4], name='x')
        self.q_target_pl = tf.placeholder(tf.float32, [None], name="Q_target")
        self.actions_pl = tf.placeholder(tf.int32, [None], name="actions")
        
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer =                 ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20,                 tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.x_pl, c_names, n_l1, w_initializer, b_initializer)
        
        gather_indices = tf.range(self.batch_size) * tf.shape(self.q_eval)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.q_eval, [-1]), gather_indices)
        with tf.variable_scope('loss'):
            self.losses = tf.squared_difference(self.q_target_pl, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)
        with tf.variable_scope('train'):
            self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        
        self.x_pl_ = tf.placeholder(tf.float32, [None, 80, 80,4], name='x')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.x_pl_, c_names, n_l1, w_initializer, b_initializer)
    
    def replace_parms(self, sess):
        t_parms = tf.get_collection('target_net_params')
        e_parms = tf.get_collection('eval_net_params')
        sess.run([tf.assign(t, e) for t, e in zip(t_parms, e_parms)])
        
    def update(self, sess, s, a, y):
        feed_dict = { self.x_pl: s, self.y_pl: y, self.actions_pl: a }
        global_step, _, loss = sess.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        return loss


# In[2]:




# In[ ]:




# In[ ]:




# In[95]:




# In[98]:




# In[ ]:



