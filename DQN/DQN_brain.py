import numpy as np
import tensorflow as tf

class DQN_network:
    # initialize the parametere
    def __init__(
            self,
            actions,
            states,
            learning_rate=0.01,
            discount=0.9,
            e_greedy=0.9,
            replace_iteration=300,
            memory_size=1000,
            batch_size=50,
            e_greedy_increasement=None
            #output_graph=False
    ):


        self.actions = actions  # a list
        self.states = states
        self.learning_rate = learning_rate
        self.gamma=discount
        self.epsilon = e_greedy
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.epsilon_increasement=e_greedy_increasement
        self.epsilon=0 if e_greedy is not None else self.esplison_max

        # count for the learning time, to determine wether to update the datas in target_net
        self.learn_step_counter=0

        # create the memory table(s,a,s_,a_)
        self.memory=np.zeros((self.memory_size,self.states*2+2))

        # build target_net and evaluate net
        self.build_net()

        # replace the parameters in target_net
        t_parameters=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'target_net_parameters')
        e_parameter=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'evaluate_net_parameters')
        # update the parameters in target_net
        self.replace_target= [tf.assign(t,e) for t,e in zip(t_parameters,e_parameter)]

        # run tensorflow operator
        self.sess = tf.Session()

        # output tensorflow document
        '''
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)
            '''

        # store all the cost, use for plotting
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # store an [s, a, r, s_]
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory, the total memory size is fixed
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # form an uniform observatio shape
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # let eval_net from all action value, and choose the largest one
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.actions)
        return action


    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


    def build_net(self):
        # store the inputs
        self.s = tf.placeholder(tf.float32, [None, self.states], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.states], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # build evaluate_net
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # build target_net
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
