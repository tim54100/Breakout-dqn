
# coding: utf-8

# In[11]:

import numpy as np
import tensorflow as tf
import os
import sys
import random
from collections import deque, namedtuple

np.random.seed(1)
tf.set_random_seed(1)


# In[39]:

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
        self.replay_memory = Memory(capacity=memory_size)
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
        #transition = [state, action, reward, n_state, done]
        self.replay_memory.store(self.Transition(state, action, reward, n_state, done))
        #self.replay_memory.store(transition)
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_parms(self.sess)
            self.saver.save(self.sess, 'saved_n/saved', global_step=self.learn_step_counter)
        tree_idx, batch_memory, ISWeights = self.replay_memory.sample(self.batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*batch_memory))
        
        q_values = self.predict(self.sess, states_batch)
        best_actions = np.argmax(q_values, axis=1)
        q_target = self.predict_t(self.sess, next_states_batch)
        targets_all_batch = q_target
        targets_all_batch[ : , action_batch.astype(np.int32)] = reward_batch +  self.gamma * q_target[np.arange(self.batch_size), best_actions]
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.gamma * q_target[np.arange(self.batch_size), best_actions]
        states_batch = np.array(states_batch)
        loss = self.update(self.sess, np.array(states_batch), np.array(action_batch), np.array(targets_batch),
                           np.array(targets_all_batch), ISWeights, tree_idx)
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
        self.q_target_all_pl = tf.placeholder(tf.float32, [None, self.n_actions], name="Q_target_all")
        self.q_target_pl = tf.placeholder(tf.float32, [None], name="Q_target")
        self.actions_pl = tf.placeholder(tf.int32, [None], name="actions")
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer =                 ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20,                 tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.x_pl, c_names, n_l1, w_initializer, b_initializer)
        
        gather_indices = tf.range(self.batch_size) * tf.shape(self.q_eval)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.q_eval, [-1]), gather_indices)
        
        with tf.variable_scope('loss'):
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target_all_pl - self.q_eval), axis=1)
            self.losses = tf.squared_difference(self.q_target_pl, self.action_predictions)
            self.loss = tf.reduce_mean(self.ISWeights * self.losses)
            
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
        
    def update(self, sess, s, a, y, y_all, ISWeights, tree_idx):
        feed_dict = { self.x_pl: s, self.q_target_pl: y, self.q_target_all_pl: y_all, self.actions_pl: a, self.ISWeights: ISWeights }
        _, abs_errors, loss = self.sess.run([self.train_op, self.abs_errors, self.loss],feed_dict)
        self.replay_memory.batch_update(tree_idx, abs_errors) # update priority
        return loss


# In[13]:

class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = []  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data.append(data)  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            self.data.pop(0)

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [], np.empty((n, 1))
        pri_seg = (self.tree.total_p() / n)       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        max_prob = np.max(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob/max_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# In[ ]:




# In[73]:

#np.array(a*4).shape


# In[62]:

#s=np.hstack(([[1,80],80,4], [1], [1], [1,80,80,4], [1]))


# In[49]:

#Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# In[60]:

#t=Transition([1,80,80,4], [1], [1], [1,80,80,4], [1])


# In[79]:

#a= [[0,1,2],[3,4,5]]


# In[97]:

b= np.array([0,1,2,3,4,5,6])


# In[107]:




# In[105]:



