# coding: utf-8

# In[11]:

import numpy as np
import tensorflow as tf
import os
import sys
import random
from collections import deque, namedtuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1)
tf.set_random_seed(1)


# In[39]:
def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.1),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg
class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        st_shape,
        learning_rate = 0.000017,
        reward_decay = 0.9,
        epsilon_start = 0.9,
        epsilon_end = 0.0001,
        replace_target_iter = 300,
        memory_size = 20000,
        batch_size = 32,
        explore = 100000,
        epsilon_decrease = True,
        training = True,
        output_graph = False,
    ):
        self.n_actions=n_actions
        self.width=int(st_shape[0])
        self.height=int(st_shape[1])
        # self.grayscale=True if st_shape[2]=='1' else False
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_decrease =  epsilon_decrease
        self.training = training
        self.epsilon_end = epsilon_end if self.training else 0
        self.epsilon_start = epsilon_start if self.training else 0
        self.epsilon = epsilon_start
        self.explore = explore
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # total_learning_step
        self.learn_step_counter = 0
        self.min_reward = -1
        self.max_reward = 1
        # initialize zero memory [state, action, reward, next_state, done]
        self.replay_memory = Memory(capacity=self.memory_size)
        #self.replay_memory = []
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        
        self._build_model()
        
        self.saver = tf.train.Saver(max_to_keep=30)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            #print checkpoint.model_checkpoint_path
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        if not os.path.exists('saved_networks'):
            os.makedirs('saved_networks')
        if output_graph:
            tf.summary.FileWriter('./logs/', self.sess.graph)

        self.cost_his = []
    
    
    def make_policy(self, state):
        action_probs =np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        q_values = self.predict(self.sess, np.expand_dims(state, 0))[0]
        #print np.ones(self.n_actions, dtype=float) * (self.epsilon/ self.n_actions )
        #print "q_values" + str(q_values)
        best_action = np.argmax(q_values)
        action_probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        #print action
        """if np.random.uniform() > self.epsilon:  # choosing action
            actions_value = self.predict(self.sess, np.expand_dims(state, 0))[0]
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)"""
        if self.epsilon > self.epsilon_end and self.epsilon_decrease:
            self.epsilon -= ((self.epsilon_start-self.epsilon_end)/self.explore)
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end
        return action

    def store_transition(self, state, action, reward, n_state, done):
        #transition = [state, action, reward, n_state, done]
        reward = max(self.min_reward, min(self.max_reward, reward))
        self.replay_memory.store(self.Transition(state, action, reward, n_state, done))
        #self.replay_memory.append(self.Transition(state, action, reward, n_state, done))
        #if len(self.replay_memory) > self.memory_size:
            #self.replay_memory.pop(0)
        #self.replay_memory.store(transition)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter ==  self.replace_target_iter - 1:
            self.replace_parms(self.sess)
            self.saver.save(self.sess, 'saved_networks/' + "saved" + '_dqn',\
                            global_step = tf.contrib.framework.get_global_step())
        tree_idx, batch_memory, ISWeights = self.replay_memory.sample(self.batch_size)
        #batch_memory = random.sample(self.replay_memory, self.batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*batch_memory))
        #print "state:"+str(states_batch[0])
        #print "action:"+str(action_batch[0])
        #print "reward:"+str(reward_batch[0])
        #print "next_states:"+str(next_states_batch[0])
        q_values = self.predict(self.sess, states_batch)
        q_values_ = self.predict(self.sess, next_states_batch)
        best_actions = np.argmax(q_values_, axis=1)
        q_next = self.predict_t(self.sess, next_states_batch)
        #print q_values
        q_target = q_values.copy()
        re_done = np.invert(done_batch).astype(np.float32) # np.array([ abs(x-1)  for x in done_batch]).astype(np.float32)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        selected_q_next = q_next[batch_index, best_actions]
        #q_target[batch_index, action_batch.astype(np.int32)] = reward_batch + self.gamma * selected_q_next
        #q_target[batch_index, action_batch.astype(np.int32)] = reward_batch + self.gamma * selected_q_next
        q_target[batch_index, action_batch.astype(np.int32)] = reward_batch + self.gamma * selected_q_next * re_done
        #target = reward_batch + self.gamma * selected_q_next * re_done
        #targets_batch = reward_batch + self.gamma * q_target[np.arange(self.batch_size), best_actions]
        #states_batch = np.array(states_batch)
        loss = self.update(self.sess, np.array(states_batch), np.array(q_values), np.array(action_batch), np.array(q_target),
                           ISWeights, tree_idx)
        #loss = self.update(self.sess, np.array(states_batch), np.array(target), np.array(action_batch), np.array(q_target),
        #                   ISWeights, tree_idx)
        #loss = self.update(self.sess, np.array(states_batch), np.array(q_values), np.array(action_batch), np.array(q_target))
        #self.cost_his.append(loss)
        
        self.learn_step_counter += 1
        return loss
        
    def predict(self, sess, state):
        #self.predictions=sess.run(self.q_eval, { self.x_pl: state, self.tf_is_training : False })
        #gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        #self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        return sess.run(self.q_eval, { self.x_pl: state, self.tf_is_training : self.training })
    
    def predict_t(self, sess, state):
        return sess.run(self.q_next, { self.x_pl_: state, self.tf_is_training : self.training })
    
    def _build_model(self):
        def build_layers(X, c_names, w_initializer, b_initializer):
            #X/=255.0
            with tf.variable_scope('l1'):
                conv1 = tf.contrib.layers.conv2d(
                    X, 32, 8, 4, activation_fn=tf.nn.relu , padding="VALID",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                #conv1 =tf.layers.dropout(conv1, rate=0.5, training=self.tf_is_training)
                #pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
                
            with tf.variable_scope('l2'):
                conv2 = tf.contrib.layers.conv2d(
                    conv1, 64, 4, 2, activation_fn=tf.nn.relu  , padding="VALID",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                #conv2 =tf.layers.dropout(conv2, rate=0.5, training=self.tf_is_training)
                #pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
                
            with tf.variable_scope('l3'):
                conv3 = tf.contrib.layers.conv2d(
                    conv2, 64, 3, 1, activation_fn=tf.nn.relu  , padding="VALID",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                #conv3 =tf.layers.dropout(conv3, rate=0.5, training=self.tf_is_training)
                #pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

            '''with tf.variable_scope('l4'):
                conv4 = tf.contrib.layers.conv2d(
                    pool3, 128, 3, 1, activation_fn=tf.nn.relu  , padding="SAME",
                    variables_collections=c_names, weights_initializer=w_initializer,
                    biases_initializer=b_initializer, reuse=None)
                #conv4 =tf.layers.dropout(conv4, rate=0.5, training=self.tf_is_training)
                pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")'''
            with tf.variable_scope('flatten'):
                flattened = tf.contrib.layers.flatten(conv3)
                '''fc1 = tf.contrib.layers.fully_connected(flattened , 512, activation_fn=tf.nn.relu, variables_collections=c_names\
                      , weights_initializer=w_initializer, biases_initializer=b_initializer)'''
                #fc1 = tf.contrib.layers.fully_connected(flattened , 512, activation_fn=None, variables_collections=c_names)
                #fc1 = parametric_relu(fc1)

            #with tf.variable_scope('Value'):
                fc1_V = tf.contrib.layers.fully_connected(flattened , 512, activation_fn=tf.nn.relu, variables_collections=c_names\
                      , weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=b_initializer)
                #fc1_V =tf.layers.dropout(fc1_V, rate=0.5, training=self.tf_is_training)
                self.V = tf.contrib.layers.fully_connected(fc1_V, 1, activation_fn=None, variables_collections=c_names\
                      , weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=b_initializer)
                #self.V =tf.layers.dropout(self.V, rate=0.5, training=self.tf_is_training)
                
            with tf.variable_scope('Advantage'):
                fc1_A = tf.contrib.layers.fully_connected(flattened , 512, activation_fn=tf.nn.relu, variables_collections=c_names\
                      , weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=b_initializer)
                #fc1_A =tf.layers.dropout(fc1_V, rate=0.5, training=self.tf_is_training)
                self.A = tf.contrib.layers.fully_connected(fc1_A, self.n_actions, activation_fn=None, variables_collections=c_names\
                      , weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=b_initializer)
                #self.A =tf.layers.dropout(self.A, rate=0.5, training=self.tf_is_training)

            #with tf.variable_scope('Q'):
                #out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)) # Q = V(s) + A(s,a)
                out = self.V + (self.A - tf.reduce_mean(self.A, reduction_indices=1, keep_dims=True)) # Q = V(s) + A(s,a)
                #gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
               # action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
                
            return out
            #return self.A
        self.x_pl = tf.placeholder(tf.float32, [None, 84, 84,4], name='x')
        self.q_value_pl = tf.placeholder(tf.float32, [None, self.n_actions], name="Q_value_pl")
        self.q_target_pl = tf.placeholder(tf.float32, [None,self.n_actions], name="Q_target")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(tf.int32, [None], name="actions")
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        self.tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing
        
        with tf.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES],\
                           tf.truncated_normal_initializer(0, 0.02)\
                           ,tf.constant_initializer(0)  #tf.random_normal_initializer(0., 0.3)\# config of layers

            self.q_eval = build_layers(self.x_pl, c_names, w_initializer, b_initializer)
        
        
        with tf.variable_scope('loss'):
            #self.abs_errors = tf.reduce_sum(tf.abs(self.q_target_pl - self.q_eval), axis=1)    # for updating Sumtree
            #self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target_pl, self.q_eval))
            
            #self.gather_indices = tf.range(self.batch_size) * tf.shape(self.q_eval)[1] + self.actions_pl
            #self.action_predictions = tf.gather(tf.reshape(self.q_eval, [-1]), self.gather_indices)
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target_pl - self.q_eval), axis=1)  # for updating Sumtree
            #self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl,  self.action_predictions))
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target_pl, self.q_eval))
            #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target_pl, self.q_eval))
            
        with tf.variable_scope('train'):
            #self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.5, 1e-6)
            #self.optimizer = tf.train.AdamOptimizer(self.lr,epsilon=1e-3)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        
        self.x_pl_ = tf.placeholder(tf.float32, [None, 84, 84,4], name='x_')
        with tf.variable_scope('target_net'):
            c_names_ = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.x_pl_, c_names_, w_initializer, b_initializer)
    
    def replace_parms(self, sess):
        t_parms = tf.get_collection('target_net_params')
        e_parms = tf.get_collection('eval_net_params')
        sess.run([tf.assign(t, e) for t, e in zip(t_parms, e_parms)])
        """update_ops = []
        for e1_v, e2_v in zip(e_parms, t_parms):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        sess.run(update_ops)"""

        
    def update(self, sess, s, y, a, target, ISWeights, tree_idx):
        #feed_dict = { self.x_pl: s, self.q_value_pl : q_v, self.q_target_pl: y, self.actions_pl: a, self.ISWeights: ISWeights,
        #              self.tf_is_training : self.training }
        feed_dict = { self.x_pl: s, self.q_target_pl: target, self.actions_pl: a, self.ISWeights: ISWeights,
                      self.tf_is_training : self.training }
        #print y
        self.global_step, _, abs_errors, loss = self.sess.run([tf.contrib.framework.get_global_step(), self.train_op, self.abs_errors\
                   , self.loss],feed_dict)
        #print a
        #print self.sess.run(self.gather_indices , feed_dict)
        #print self.sess.run(self.q_eval , feed_dict)
        #print np.array(abs_errors)/np.max(np.array(abs_errors))
        self.replay_memory.batch_update(tree_idx, abs_errors) # update priority
        return loss
    '''def update(self, sess, s, q_v, a, y):
        feed_dict = { self.x_pl: s, self.q_target_pl: y, self.actions_pl: a, self.tf_is_training : self.training }
        #feed_dict = { self.x_pl: s, self.y_pl: y, self.actions_pl: a, self.tf_is_training : self.training }
        self.global_step, _, loss = self.sess.run([tf.contrib.framework.get_global_step(), self.train_op, self.loss],feed_dict)

        return loss'''


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
        pointer = self.data_pointer % self.capacity
        tree_idx = pointer + self.capacity - 1
        if self.data_pointer < self.capacity:
            self.data.append(data)  # update data_frame
        else:  # replace when exceed the capacity
            #self.data_pointer = 0
            #print "len(data): "+str(len(self.data))
           # print "tree_idx: "+str(tree_idx)
            self.data[pointer] = data
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        #if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            #self.data_pointer = 0
        #    self.data.pop(self.data_pointer % self.capacity)

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
        #print "total:"+str(self.tree.total_p())+", n:"+str(n)
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            #print "a:"+str(a)+", b:"+str(b)+", pri_seg:"+str(pri_seg) + ", i: " + str(i)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        #print ps
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


