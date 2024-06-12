import os
import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
import time

rng = np.random.default_rng()

class Memory(object):
    def __init__(self, mem_size, input_shape, nr_actions):

        # Constants
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.nr_actions = nr_actions

        # Counter
        self.mem_cntr = 0

        # Values
        self.state_memory = np.zeros((self.mem_size, self.input_shape))
        self.new_state_memory = np.zeros((self.mem_size, self.input_shape))
        self.action_memory = np.zeros((self.mem_size, self.nr_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, new_state, done):

        # Index
        index = self.mem_cntr

        # Storing
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)

        # Update
        self.mem_cntr += 1

    def generate(self):

        states = self.state_memory
        new_states = self.new_state_memory
        rewards = self.reward_memory
        actions = self.action_memory
        terminal = self.terminal_memory

        return states, actions, rewards, new_states, terminal

class PPO(object):
    def __init__(self, GAMMA, LAM, EPSILON, C_LR, A_LR, input_dims, action_dims, nr_actions,  
        actor_dims, critic_dims, EPOCHs, max_size=32, chkpt_dir='tmp'):

        # Constants
        self.GAMMA = GAMMA
        self.lam = LAM
        self.epochs = EPOCHs
        self.nr_actions = nr_actions
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.adims1, self.adims2 = actor_dims
        self.cdims1, self.cdims2 = critic_dims
        self.max_size = max_size

        self.checkpoint_file =  os.path.join(chkpt_dir, f'{int(time.time())}_ppo.ckpt')
        
        self.memory = Memory(self.max_size, self.input_dims, self.nr_actions)


        # Session and Placeholders
        self.sess = tf.Session()

        self.input = tf.placeholder(tf.float32, shape=(None, input_dims), name = 'input')

        self.disc_reward = tf.placeholder(tf.float32, shape=(None, self.nr_actions), name='discounted_reward')

        self.action = tf.placeholder(tf.float32, shape=(None, self.nr_actions), name='actions')


        # Critic network & Advantage Estimate
        self.v = self.build_critic('critic')

        self.advantage = self.disc_reward - self.v


        # Actor network 
        probs = self.build_actor('probs', trainable=True)

        old_probs = self.build_actor('old_probs', trainable=False)


        # Distribution and Sampling
        old_pi = tfp.distributions.Categorical(probs = old_probs)

        self.sample_action = tf.squeeze(old_pi.sample())


        # Parameters & Update
        pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='probs')

        oldpi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_probs')

        self.update_oldpi = [oldpi.assign(pi) for oldpi, pi in zip(oldpi_params, pi_params)]


        # Train - Critic
        self.closs = tf.reduce_mean(tf.square(self.advantage)) # squaure ---> mean

        self.ctrain = tf.train.AdamOptimizer(C_LR).minimize(self.closs)


        # Train - Actor
        self.adv = tf.placeholder(tf.float32, shape=(None, self.nr_actions), name='advantage')

        log_prob_p =[]
        log_prob_op = []
        a = tf.squeeze(self.action)

        for i in range(self.max_size):
            pi = tfp.distributions.Categorical(probs=probs[i])
            opi = tfp.distributions.Categorical(probs=old_probs[i])

            log_prob_p.append(pi.log_prob(a[i]))
            log_prob_op.append(opi.log_prob(a[i]))

        log_prob_p = tf.convert_to_tensor(log_prob_p)
        log_prob_op = tf.convert_to_tensor(log_prob_op)

        diff = log_prob_p - log_prob_op
        
        ratio = tf.exp(diff)
    
        gae = tf.squeeze(self.adv)

        surr_loss = ratio*gae

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr_loss, 
            tf.clip_by_value(ratio, 1-EPSILON, 1+EPSILON)*gae)) - 0.01*tf.reduce_mean(old_pi.entropy())
    
        self.atrain = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())

    def build_actor(self, name, trainable):
        
        dense1 = tf.layers.dense(self.input, units=self.adims1, activation='tanh', trainable=trainable)
        dense2 = tf.layers.dense(dense1, units=self.adims2, activation='tanh', trainable=trainable)

        out = tf.layers.dense(dense2, units=self.action_dims, activation='softmax', trainable=trainable)

        return out

    def build_critic(self, name):
        
        dense1 = tf.layers.dense(self.input, activation='relu', units=self.cdims1)
        dense2 = tf.layers.dense(dense1, activation='relu', units=self.cdims2)

        v = tf.layers.dense(dense2, units=1)

        return v

    def learn(self):

        if self.memory.mem_cntr < self.max_size:
            return
        else:

            state, action, reward, new_state, done = self.memory.generate()

            v = self.sess.run(self.v, feed_dict={self.input:new_state})

            dr = []
            for i in range(self.max_size):
                disc_r = reward[i] + self.GAMMA*v[i]*done[i]
                dr.append(disc_r)
            dr = np.reshape(dr, (self.max_size, 1))  

            adv = self.sess.run(self.advantage, feed_dict={self.input:state, self.disc_reward:dr})

            coeff = self.GAMMA*self.lam
            gae = np.empty((self.max_size, 1))
            for t in range(self.max_size):
                ad = 0
                for i in range(0, self.max_size-t):
                    s = (adv[i+t])*(coeff**i)
                    ad += s
                gae[t] = ad

            gae = (gae - np.mean(gae)) / np.std(gae) 

            [self.sess.run(self.atrain, feed_dict={self.input:state, self.adv:gae, 
                self.action:action}) for _ in range(self.epochs)]

            [self.sess.run(self.ctrain, feed_dict={self.input:state, 
                self.disc_reward:dr}) for _ in range(self.epochs)]

            self.sess.run(self.update_oldpi)

            self.memory = Memory(self.max_size, self.input_dims, self.nr_actions)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_action, feed_dict={self.input:state})

        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_model(self):
        print('...saving model...')
        tf.train.Saver().save(self.sess, self.checkpoint_file)
