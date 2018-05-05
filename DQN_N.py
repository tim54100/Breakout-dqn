import numpy as np
import os
import tensorflow as tf
import random

np.random.seed(1)
tf.set_random_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class DeepQNetwork:
    # initialize some variable, model and check the checkpoint file exist
    # if (exist) load the checkpoint file to continue to train or show
    # else  it means it will train wiithout experience 
    def __init__(
        self,
        n_actions,
        st_shape=[84,84],
        learning_rate = 0.00017,
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
        self.epsilon = self.epsilon_start
        self.explore = explore
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # total_learning_step
        self.min_reward = -1
        self.max_reward = 1
        # initialize zero memory [state, action, reward, next_state, done]
        #self.replay_memory = Memory(capacity=self.memory_size)
        self.replay_memory = []
        #self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        
        self._build_model()
        
        self.saver = tf.train.Saver(max_to_keep=1)
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
        
        self.learn_step_counter = tf.train.global_step(self.sess, self.global_step)
        self.cost_his = []

    # It wiil choose the biggest q_value to be the possible choice
    # but if (epsilon not equal = 0) then we still possibly choose other choice
    # and the probability is defined by the epsilon and actions amount
    def make_policy(self, state):
    	q_values = self.predict(np.expand_dims(state, 0))[0]
    	action_probs = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
    	best_action = np.argmax(q_values)
        action_probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    	if self.epsilon_decrease and self.epsilon > self.epsilon_end:
    		self.epsilon = self.epsilon - ((self.epsilon_start - self.epsilon_end) / self.explore)
    		if self.epsilon < self.epsilon_end:
    		    self.epsilon = self.epsilon_end
    	return action

    # let state, action(the state chose), reward(get reward form state to n_state with action)
    # n_state(next state by state )
    def store(self, state, action, reward, n_state, done):
    	self.replay_memory.append([state, action, reward, n_state, done])
    	if len(self.replay_memory) > self.memory_size:
    	    self.replay_memory.pop(0)

    # predict the q_value with state and q_eval 
    def predict(self, state):
    	return self.sess.run(self.q_eval, {self.x_pl: state})

    # predict the q_value with state and q_target
    # q_target is the q_eval's past
    def predict_t(self, state):
    	return self.sess.run(self.q_target, {self.x_pl_: state})

    # bulid the q_eval, q_target, how to calculate loss, optimizer
    # and how to optimize q_eval
    def _build_model(self):
    	# build layers for q_eval, q_target
        def _build_layers(self, x , scope, w_init, b_init):
            with tf.variable_scope(scope):
                with tf.variable_scope("conv1"):
                    conv1 = tf.layers.conv2d(x, 32, [8, 8], [4, 4], activation=tf.nn.selu, kernel_initializer=w_init,\
	 		                                 bias_initializer=b_init)
                    #pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
                    #pool1 = tf.layers.average_pooling2d(conv1, [2, 2], [2, 2])
                with tf.variable_scope("conv2"):
                    conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2], activation=tf.nn.selu, kernel_initializer=w_init,\
	 		                                 bias_initializer=b_init)
                    #pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])
                    #pool2 = tf.layers.average_pooling2d(conv2, [2, 2], [2, 2])
                with tf.variable_scope("conv3"):
                    conv3 = tf.layers.conv2d(conv2, 64, [3, 3], activation=tf.nn.selu, kernel_initializer=w_init,\
	 		                                 bias_initializer=b_init)
                    #pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2])
                    #pool3 = tf.layers.average_pooling2d(conv3, [2, 2], [2, 2])
                '''with tf.variable_scope("conv4"):
                    conv4 = tf.layers.conv2d(conv3, 128, [3, 3], activation=tf.nn.selu, kernel_initializer=w_init,\
	 		                                 bias_initializer=b_init)
                    #pool4 = tf.layers.max_pooling2d(conv4, [2, 2], [2, 2])
                    #pool4 = tf.layers.average_pooling2d(conv4, [2, 2], [2, 2])
                with tf.variable_scope("conv5"):
                    conv5 = tf.layers.conv2d(conv4, 256, [3, 3], activation=tf.nn.selu, kernel_initializer=w_init,\
	 			bias_initializer=b_init)
                    #pool5 = tf.layers.max_pooling2d(conv5, [2, 2], [2, 2])
                with tf.variable_scope("conv6"):
                    conv6 = tf.layers.conv2d(conv5, 512, [3, 3], activation=tf.nn.selu, kernel_initializer=w_init,\
	 			bias_initializer=b_init)'''
                with tf.variable_scope("flatten"):
                    flattend = tf.layers.flatten(conv3)
                    fc = tf.layers.dense(flattend, 512, activation=tf.nn.selu,\
	 		                             kernel_initializer=tf.random_normal_initializer(stddev=0.02), bias_initializer=b_init)
                    '''batch_norm = tf.contrib.layers.batch_norm(fc, decay=0.99, updates_collections=None, epsilon=1e-5,\
                                                 scale=True, is_training=True, scope="bn")
                    batch_norm = tf.nn.selu(batch_norm)'''
                with tf.variable_scope("advantage"):
                    a = tf.layers.dense(fc, self.n_actions, activation=None,\
	 		                            kernel_initializer=tf.random_normal_initializer(stddev=0.02), bias_initializer=b_init)
                with tf.variable_scope("value"):
                    v = tf.layers.dense(fc, 1, activation=None,\
	 		                            kernel_initializer=tf.random_normal_initializer(stddev=0.02), bias_initializer=b_init)
                out = v + (a - tf.reduce_mean(a, reduction_indices=1, keep_dims=True)) # Q = V(s) + A(s,a)
                #out = tf.contrib.layers.batch_norm(out, decay=0.99, updates_collections=None, epsilon=1e-5,\
                #                                 scale=True, is_training=True, scope="bn")

                return out
        self.x_pl = tf.placeholder(tf.float32, [None, self.width, self.height,4], name='x')
        self.y_pl = tf.placeholder(tf.float32, [None, self.n_actions], name="y")
        self.x_pl_ = tf.placeholder(tf.float32, [None, self.width, self.height,4], name='x_')
        with tf.variable_scope('DQN'):
        	scope_name, w_init, b_init = 'eval_net', tf.truncated_normal_initializer(0, 0.02), tf.constant_initializer(0.01)
        	scope_name_ = 'target_net'
        	self.q_eval = _build_layers(self, self.x_pl, scope_name, w_init, b_init)
        	self.q_target = _build_layers(self, self.x_pl_, scope_name_, w_init, b_init)

        with tf.variable_scope('loss'):
        	self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.q_eval))
        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    # let q_target equal to q_eval
    def replace_parms(self):
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DQN/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DQN/target_net')
        self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])

    # if learn fixed times we will ues replace_parms function and save the model
    # get batch's size memory from we store before
    # then use them to calculate q_values(new) and delivery to update
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter ==  self.replace_target_iter - 1:
            self.replace_parms()
            self.saver.save(self.sess, 'saved_networks/' + "saved" + '_dqn',\
                        global_step = tf.contrib.framework.get_global_step())
        self.learn_step_counter+=1
        batch_memory = random.sample(self.replay_memory, self.batch_size)
        states_batch, actions_batch, rewards_batch, n_states_batch, dones_batch = map(np.array, zip(*batch_memory))
        q_values = self.predict(states_batch)
        q_values_ = self.predict(n_states_batch)
        best_actions = np.argmax(q_values_, axis=1)
        q_next = self.predict_t(n_states_batch)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_values[batch_index, actions_batch] = rewards_batch + np.invert(dones_batch).astype(np.float32) *\
                                               self.gamma * q_next[batch_index, best_actions]
        loss = self.update(states_batch, q_values)
        return loss
    # feed the y_pl(q_values(new)) and states to calculate loss, then use loss to optimize q_eval
    def update(self, states, y_pl):
        feed_dict={self.x_pl: states, self.y_pl: y_pl}

        self.global_step, _, loss = self.sess.run([tf.train.get_global_step(), self.train_op, self.loss],feed_dict)

        return loss
