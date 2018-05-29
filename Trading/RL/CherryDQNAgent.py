'''
Cherry:
1. Split initial_cash_supply to 2 parts, coin and cash. Note: in utils_v3.py, portfolio.starting_cash is actually starting_total_value. We do not change the name.
2. epsilon_decay defaults to 0.996
3. In train(), default start_time=datetime.datetime(2017,1,1,0), end_time=datetime.datetime(2018,2,1,0)
4. Choose memory_system on initialization.
'''

from importlib import reload
import utils_v3
reload(utils_v3)
from utils_v3 import *

import tensorflow as tf

import random
import numpy as np
# from collections import deque
import heapq
import pandas as pd

import datetime


# Neural Network for the Q value approximation
class QValue_NN:
    def __init__(self, n_external, n_internal, n_action=3, seq_len=10):
        self.n_external = n_external
        self.n_internal = n_internal
        self.n_action = n_action
        self.seq_len = seq_len
        self.__build_model()
        
    def __huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        #error = prediction - target
        #return np.mean(np.sqrt(1+np.square(error))-1, axis=-1)
        err2 = tf.squared_difference(target, prediction)
        return tf.reduce_mean(tf.square(1+err2)-1, axis=-1)

    def __build_model(self): 
        
        # To be tund
        self.n_neurons = 100 
        self.n_lstm_layers = 2 
        self.fc1_size = 20 
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.learning_rate = tf.placeholder(tf.float32, []) 
        self.max_gradient_norm = 5  
        
        with tf.variable_scope("LSTM", initializer=tf.contrib.layers.xavier_initializer()):
            self.X_external = tf.placeholder(tf.float32, [None, self.seq_len, self.n_external])
            self.X_internal = tf.placeholder(tf.float32, [None, self.n_internal])
            self.y = tf.placeholder(tf.float32, [None, self.n_action])
            
            layers = [tf.contrib.rnn.LSTMCell(num_units=self.n_neurons, \
                                              initializer=tf.contrib.layers.xavier_initializer(), \
                                              activation=tf.nn.elu)
                     for layer in range(self.n_lstm_layers)]
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
            
            LSTM_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, self.X_external, dtype=tf.float32) # [batch_size, seq_len, n_neurons]
            LSTM_outputs = tf.nn.dropout(LSTM_outputs, self.keep_prob) # dropout after LSTM
            LSTM_outputs = LSTM_outputs[:, -1, :] # [batch_size, n_neurons]
            fc1 = tf.layers.dense(LSTM_outputs, self.fc1_size, activation=tf.nn.leaky_relu) # first FC, [batch_size, fc1_size]
            fc1 = tf.nn.dropout(fc1, self.keep_prob) # dropout after first FC
            all_states = tf.concat([fc1, self.X_internal], axis=1) # [batch_size, fc1_size+n_internal]
            self.preds = tf.layers.dense(all_states, self.n_action, activation=tf.nn.leaky_relu)  # second FC #[batch_size, n_action]
            self.loss = tf.nn.l2_loss(self.preds - self.y)
            #self.loss = self.__huber_loss(self.y, self.preds)
            
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) 
            #self.train_op = optimizer.minimize(self.loss)
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def train(self, session, external_states, internal_states, qvalues, keep_prob=0.8, lr=0.01):
        _, loss = session.run([self.train_op, self.loss], 
                              feed_dict={self.X_external: external_states, 
                                         self.X_internal: internal_states, 
                                         self.y:qvalues,
                                         self.keep_prob: keep_prob,
                                         self.learning_rate: lr})
        return loss

    def predict(self, session, state, keep_prob=0.8):
        states = np.array(state) # list2array
        external_states = states[:-self.n_internal].reshape([self.seq_len, -1]) # extract external states and reshape -> [self.seq_len, self.n_external]
        external_states = np.expand_dims(external_states, axis=0) # reshape -> [1, self.seq_len, self.n_external] 
        internal_states = states[-self.n_internal:] # extract internal states
        internal_states = np.expand_dims(internal_states, axis=0) # reshape -> [1, self.n_internal] 
        return session.run(self.preds, feed_dict={self.X_external: external_states,
                                                  self.X_internal: internal_states,
                                                  self.keep_prob: keep_prob})  

# Agent Implementation
class CherryDQNAgent:
    
    # initialize internal variables
    def __init__(self, initial_total_value, memory_system, coin_name='BTC', num_coins_per_order=1.0, recent_k=0,
                 gamma=0.95, epsilon_min=0.005, epsilon_decay=0.996, 
                 seq_len=10, external_states=external_state_list, internal_states=internal_state_list, 
                 verbose=False):
        
        self.initial_total_value = initial_total_value
        
        self.max_mem_len = 100000
        self.memory = [] 
        
        self.batch_size = 50
        self.gamma = gamma
        self.epsilon=1.0
        self.epsilon_min=epsilon_min 
        self.epsilon_decay=epsilon_decay
        
        self.coin_name = coin_name
        
        # External states & env
        self.external_states = external_states
        self.env = Environment(coin_name=coin_name, states=external_states, recent_k=recent_k)
        
        # Internal states & portfolio
        self.internal_states = internal_states
        self.portfolio = Portfolio(cash_supply=initial_total_value, num_coins_per_order=num_coins_per_order, states=internal_states,
                                   verbose=verbose, final_price=self.env.getFinalPrice())

        print('epsilon_decay = ', epsilon_decay)
        
        # NN model
        tf.reset_default_graph()
        self.seq_len = seq_len # how many steps the agent sees before making a decision, used in env.getStateSequence
        with tf.variable_scope("model", initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.model = QValue_NN(n_external=self.env.getStateSpaceSize(), 
                                   n_internal=self.portfolio.getStateSpaceSize(), 
                                   n_action=self.portfolio.getActionSpaceSize(),
                                   seq_len=self.seq_len)
            self.old_vars = {v.name.split('model/')[-1] : v for v in tf.trainable_variables() if v.name.startswith(scope.name + "/")}
        with tf.variable_scope("target_model", initializer=tf.contrib.layers.xavier_initializer()):
            self.target_model = QValue_NN(n_external=self.env.getStateSpaceSize(), 
                                   n_internal=self.portfolio.getStateSpaceSize(), 
                                   n_action=self.portfolio.getActionSpaceSize(),
                                   seq_len=self.seq_len)
            
#         self.state_mean = None
        self.train_cum_returns = []
        
        self.test_cum_returns = []
        self.test_portfolio_values = []
        self.test_portfolio_values_cash = []
        self.test_portfolio_values_coin = []
        self.test_actions = []
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=2)
        
        self.sample_prob_list = np.arange(self.max_mem_len)**0.2
        self.sample_prob_list = self.sample_prob_list / np.sum(self.sample_prob_list)
        
        self.memory_system = memory_system
        
     
    def plot_external_states(self):
        self.env.plot(self.external_states)
    
    def __act(self, session, state):
        if np.random.rand() < self.epsilon:
            return random.choice(list(Action))
        act_values = self.model.predict(session, state)
        action_intended = Action(np.argmax(act_values[0]))
        L = np.argsort(-act_values[0])
        if str(action_intended) == 'Action.SELL' and self.portfolio.portfolio_coin == 0:
            return Action(L[1])
        elif str(action_intended) == 'Action.BUY' and self.portfolio.portfolio_cash == 0:
            return Action(L[1])
        else:
            return action_intended
        
    def __update_target_model(self, sess):
        with tf.variable_scope("target_model", initializer=tf.contrib.layers.xavier_initializer()) as scope:
            assignments = [v.assign(self.old_vars[v.name.split('model/')[-1]]) \
                           for v in tf.trainable_variables() if v.name.startswith(scope.name + "/")]
            sess.run(assignments)

    def print_my_memory(self):
        mem = list(self.memory)
        mem_str = []
        for s, a, r, s_, donzo in mem:
            mem_str += ["%s_%s_%s_%s_%s" % (str(s), str(a), str(r), str(s_), str(donzo))]
    
        uniques = list(set(mem_str))
        uniques.sort() 
        
        for elem in uniques:
            print(elem)
            print(mem_str.count(elem))
            print("\n")
        
    def __replay(self, session, reward_func):            
        
        external_states_stacked = []
        internal_states_stacked = []
        qvalues_stacked = []
        
        print("Memory length:", len(self.memory))
        
#         v_end = self.memory[-1][-1]
#         v_0 = self.memory[0][-2]
#         reward = v_end / v_0 * 100
        
        batch_size = self.batch_size 
        minibatch = []
        
        if self.memory_system == 'reward_priority' or self.memory_system == 'abs_reward_priority':
            idx_list = np.arange(len(self.memory))
            plist = self.sample_prob_list[:len(self.memory)]
            plist /= np.sum(plist)
            while (len(minibatch) < self.batch_size):
                idx = np.random.choice(idx_list, p=plist)
                minibatch.append(self.memory[idx])
        else:    
            minibatch = random.sample(self.memory, self.batch_size)
        
        for priority, record in minibatch:
            state, action, next_state, is_last_step, v_t, v_t_plus1 = record

            if reward_func == 'Andy':
                reward = (v_t_plus1 / self.initial_total_value - 1) * 100 # immediate reward, first term in Bellman Eq
            else:
                reward = (v_t_plus1 / v_t - 1) * 100
#             print('immediate reward', reward)
            
            qvalues = self.model.predict(session, state) 
#             print('target predict before action:', target)

            if is_last_step:
                qvalues[0][action.value] = reward
            else:
                a = self.model.predict(session, next_state)[0]
                t = self.target_model.predict(session, next_state)[0]
                # should be a list of 3 numbers
#                 print('t', t)
                
                # Bellman Equation
                qvalues[0][action.value] = 0 + self.gamma * t[np.argmax(a)] # one action, not sum of all actions
                if reward_func != 'Andy':
                    qvalues[0][action.value] += reward
                
#                 print('action:',action)
#                 print('target predict after action:', target)
#                 print('======')
            
            states = np.array(state) # list2array
            n_internal = len(self.internal_states) 
            external_states = states[:-n_internal] # extract external states 
            internal_states = states[-n_internal:] # extract internal states
            external_states_stacked.append(external_states)
            internal_states_stacked.append(internal_states) 
            qvalues_stacked.append(qvalues[0])

            if len(qvalues_stacked)== batch_size:
                external_states_stacked = np.vstack(external_states_stacked).reshape([batch_size, self.seq_len, -1]) # [batch_size, seq_len, n_external]
                internal_states_stacked = np.vstack(internal_states_stacked) # [batch_size, n_internal]
                qvalues_stacked = np.array(qvalues_stacked) # [batch_size, n_action]
#
                regression_loss = self.model.train(session, 
                                                   external_states_stacked, internal_states_stacked, 
                                                   qvalues_stacked)
                print('regression loss: ', regression_loss)
                
                external_states_stacked = []
                internal_states_stacked = []
                qvalues_stacked = []
        
        # update the epsilon to gradually reduce the random exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
#         self.memory = [] # clear memory after using this episode

    def preprocess(self, state, episode_beginning_price): 
        # Andy said normalizing price such that price at the beginning of an episode is 1
        for step in range( (len(state)-len(internal_state_list))//len(external_state_list) ):
            state[step*len(external_state_list):step*len(external_state_list)+5] /= episode_beginning_price
        state[-1] /= episode_beginning_price # normalize last_buy_price 
            
    # Agent Training    
    def train(self, experiment_name, session, start_time=datetime.datetime(2017,1,1,0), end_time=datetime.datetime(2018,2,1,0), episode_len=100, num_episodes=100, verbose=True, auto_save_and_load=True, save_every=50, test_every=1000, reward_func='Andy'):
        
        if auto_save_and_load:
            print('Auto loading is on, looking for saved checkpoints...')
            ckpt = tf.train.get_checkpoint_state(experiment_name)
            v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
            if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
                self.saver.restore(session, ckpt.model_checkpoint_path)
                print("Start from saved checkpoint...")
                current_episode = self.global_step.eval(session=session) + 1 # get last global_step
                print("Start from iteration:", current_episode)

            else:
                print('There is not saved parameters. Creating model with fresh parameters.')
                session.run(tf.global_variables_initializer())
                current_episode = 0

        else:
            print('Auto loading is off. Creating model with fresh parameters.')
            session.run(tf.global_variables_initializer())

        self.train_cum_returns = []
        
        if start_time is None:
                start_time = self.env.start_index
        
        n_days = (end_time - start_time) // (self.env.time_delta * 24)
        print('Training, randomly selecting episodes from ', start_time, ' to', end_time, ': ', '~', n_days, 'days\n')
        
        for i in range(current_episode, num_episodes):
            
            self.env.reset()
            self.portfolio.reset()
            temp_memory = []
            
            # randomly choose episode start time
            int_idx_start = (start_time - self.env.start_index) / self.env.time_delta
            int_idx_end = (end_time - self.env.start_index) / self.env.time_delta - (episode_len - 1)
            int_idx = random.randint(int_idx_start, int_idx_end)
            episode_start_time = self.env.df.index[int_idx]
            episode_end_time = episode_start_time + (episode_len - 1) * self.env.time_delta
            
            if verbose:
                print('====================================\n Starting episode from: ', episode_start_time) 
            
            self.env.set_current_time(episode_start_time)
            episode_beginning_price = self.env.getCurrentPrice() # needed for normalizing state features
            
            self.portfolio.portfolio_cash = self.initial_total_value/2
            self.portfolio.portfolio_coin = self.initial_total_value/2/self.env.getCurrentPrice()
            
            self.portfolio.state_dict["steps_left_in_episode"] = episode_len - 1
            self.portfolio.state_dict["last_buy_price"] = episode_beginning_price
            
            state = self.env.getStatesSequence(self.seq_len) + self.portfolio.getStates() # list
            self.preprocess(state, episode_beginning_price) # normalize price features

            # walk through A RANDOM SEQUENCE OF HOURS (an episode)
            # obtain action based on state values using the Neural Network model
            # collect reward
            # update the experience in Memory
            
            while (True): # while not the end of episode, just trade and observe
                if verbose:
                    print('Current time:', self.env.current_index)
                    
                #print(steps_left)
                    
                value_before_action = self.portfolio.getCurrentValue(self.env.getCurrentPrice())
                
                action = self.__act(session, state)
                
                v_t = self.portfolio.getCurrentValue(self.env.getCurrentPrice()) # ptfl value before action AND env.step !!!
                
                is_last_step, next_state = self.env.step(episode_end_time) # episode end time, not env end time!
                
                action = self.portfolio.apply_action(self.env.getCurrentPrice(), action, verbose)
                
                if str(action) == 'Action.BUY':
                    self.portfolio.state_dict["last_buy_price"] = self.env.getCurrentPrice()
                    #print("Last buy price changes to:", self.portfolio.state_dict["last_buy_price"])
                
                v_t_plus1 = self.portfolio.getCurrentValue(self.env.getCurrentPrice()) # ptfl value after action
                
                self.portfolio.state_dict["steps_left_in_episode"] -= 1
                
                next_state = self.env.getStatesSequence(self.seq_len) + self.portfolio.getStates() # list
                
                self.preprocess(next_state, episode_beginning_price)
                
                if verbose:
                    print(action_applied, v_t)
                    print()
                
                temp_memory.append([state, action, next_state, is_last_step, v_t, v_t_plus1])
                
                state = next_state # list
                
                if is_last_step:
                    break
            
            ep_reward = (v_t_plus1 / self.initial_total_value - 1) * 100
            
            if self.memory_system == 'abs_reward_priority':
                for record in temp_memory:
                    heapq.heappush(self.memory, (abs(ep_reward), record)) # differs from the first Blackberry
                while len(self.memory) > self.max_mem_len:
                    heapq.heappop(self.memory)
            elif self.memory_system == 'reward_priority':
                for record in temp_memory:
                    heapq.heappush(self.memory, (ep_reward, record))
                while len(self.memory) > self.max_mem_len:
                    heapq.heappop(self.memory) 
            elif self.memory_system == 'long_deque':
                for record in temp_memory:
                    self.memory.append((0, record))
                while len(self.memory) > self.max_mem_len:
                    self.memory.pop(0)
            else:
                raise ValueError('unsupported memory system')
           
            # train the Neural Network incrementally with the new experiences
#             if len(self.memory) > self.batch_size:
#                 self.__replay(session, self.batch_size)
            if len(self.memory) > self.batch_size:
                self.__replay(session, reward_func)
            self.__update_target_model(session)

            cum_return = self.portfolio.getReturnsPercent(self.env.getCurrentPrice())
            self.train_cum_returns.append(cum_return)

            print("episode: {}/{}, returns: {:.2}, epsilon: {:.2}"
                  .format(i+1, num_episodes, 
                          cum_return, 
                          self.epsilon))
            
            if auto_save_and_load and (i+1) % save_every == 0:
                self.global_step.assign(i).eval(session=session)
                path = self.saver.save(session, "./"+experiment_name+"/model.ckpt", global_step=self.global_step)
                print('saved to ' + path)
                
            if (i+1) % test_every == 0:
                test_start = datetime.datetime(2018,2,1,0)
                test_end = datetime.datetime(2018,4,1,0)
                _ = self.test(session, start_time = test_start, end_time = test_end, episode_len=20, verbose=False)
                                
    ### Sample Usage:
        
    ### import datetime
    ### start = datetime.datetime(2018,1,1,0)
    ### agent.test(start_time = start)
    
    def test(self, session, start_time, end_time=None, episode_len = 20, verbose=True, print_freq='daily', xaction_fee=0.25/100):
            print('Testing, setting epsilon to zero...')
            epsilon_temp = self.epsilon
            self.epsilon = 0.000001
            
            if end_time is None or end_time >= self.env.end_index:
                end_time = self.env.end_index - self.env.time_delta
                
            n_days = (end_time - start_time) // (self.env.time_delta * 24)
            print('Testing, going through all possible episodes in range ', start_time, ' to', end_time, ': ', '~', n_days, 'days\n')
            print('Actually, because there are so many episodes, we choose 1 starting point every 100 hours...')

            int_idx_start = int((start_time - self.env.start_index) / self.env.time_delta)
            int_idx_end = int((end_time - self.env.start_index) / self.env.time_delta - (episode_len - 1))
            
            pct_return_list = []
            test_history_list = []
            
            for int_idx in range(int_idx_start, int_idx_end+1, 100):
                self.env.reset()
                self.portfolio.reset()

                episode_start_time = self.env.df.index[int_idx]
                episode_end_time = episode_start_time + (episode_len - 1) * self.env.time_delta

#                 if verbose:
                print('====================================\n Starting episode from: ', episode_start_time) 

                self.env.set_current_time(episode_start_time)
                episode_beginning_price = self.env.getCurrentPrice() # needed for normalizing state features

                self.portfolio.state_dict["steps_left_in_episode"] = episode_len - 1
                self.portfolio.state_dict["last_buy_price"] = episode_beginning_price

                state = self.env.getStatesSequence(self.seq_len) + self.portfolio.getStates() # list
                self.preprocess(state, episode_beginning_price) # normalize price features

                # walk through A RANDOM SEQUENCE OF HOURS (an episode)
                # obtain action based on state values using the Neural Network model
                # remember episode history
                start_day = episode_start_time.day
                verbose_g = verbose

                episode_action_history = []
                episode_ptfl_history_cashval = []
                episode_ptfl_history_coinval = []
                episode_ptfl_history_totalval = []
                
                self.portfolio.state_dict["steps_left_in_episode"] = episode_len - 1
                self.portfolio.state_dict["last_buy_price"] = episode_beginning_price
                
                while (True): # while not end of episode

                    if verbose:
                        if print_freq == 'hourly':
                            print('Current time:', self.env.current_index)
                            verbose = True
                        if print_freq == 'daily':
                            if self.env.current_index.hour == 0:
                                print('Current time:', self.env.current_index)
                                verbose = True
                            else:
                                verbose = False
                        elif print_freq == 'weekly': 
                            if self.env.current_index.day in np.roll((np.arange(28)+1), 28-start_day+1)[::7] \
                                      and self.env.current_index.hour == 0:
                                print('Current time:', self.env.current_index)
                                verbose = True
                            else:
                                verbose = False


                    action = self.__act(session, state)
                    isDone, next_state = self.env.step(episode_end_time) # order changed, and it's episode end time
                    action = self.portfolio.apply_action(self.env.getCurrentPrice(), action, verbose, xaction_fee)
                    
                    if str(action) == 'Action.BUY':
                        self.portfolio.state_dict["last_buy_price"] = self.env.getCurrentPrice()
                        
                    self.portfolio.state_dict["steps_left_in_episode"] -= 1

                    next_state = self.env.getStatesSequence(self.seq_len) + self.portfolio.getStates()
                    self.preprocess(next_state, episode_beginning_price)
                    state = next_state
               
                    episode_ptfl_history_cashval.append(self.portfolio.portfolio_cash)
                    episode_ptfl_history_coinval.append(self.portfolio.portfolio_coin * self.env.getCurrentPrice())
                    episode_ptfl_history_totalval.append(episode_ptfl_history_cashval[-1] + episode_ptfl_history_coinval[-1])

                    episode_action_history.append(action.value)

                    verbose = verbose_g

                    if isDone:
                        break
                    # end for each step

                episode_percent_return = self.portfolio.getReturnsPercent(self.env.getCurrentPrice())
                pct_return_list.append(episode_percent_return)
#                 if verbose:
                print('Episode percentage return:', episode_percent_return)
                    
                test_history_list.append([episode_start_time, episode_end_time, episode_action_history, \
                                         [episode_ptfl_history_cashval, episode_ptfl_history_coinval, episode_ptfl_history_totalval] ])
                # end for each episode
            
            print('Average percentage return over all tests:', np.mean(pct_return_list))
            self.epsilon = epsilon_temp
            return test_history_list
        
    def plot_action(self, start_time, end_time=None):
        import matplotlib.pyplot as plt
        
        if end_time is None: # default: one day
            end_time = start_time + self.env.time_delta * 24
        
        df = self.env.df
        df = df.loc[df.index >= start_time]
        df = df.loc[df.index <= end_time]
        prices = df['USDT_BTC_open']
        
        actions = self.test_actions
        actions = actions[actions.index >= start_time]
        actions = actions[actions.index < end_time]
        
        fig, ax1 = plt.subplots(figsize = (15, 8))
        
        ax1.plot(prices.index, prices, 'b-')
        ax1.set_ylabel('Price', color='b', fontsize=15)
        ax1.tick_params('y', colors='b', labelsize=15)
    
        hold = actions[actions == 1]
        buy = actions[actions == 2]
        sell = actions[actions == 0]
        
        ax2 = ax1.twinx()
        ax2.scatter(hold.index, hold, c='blue', label='HOLD')
        ax2.scatter(buy.index, buy, c='green', label='BUY')
        ax2.scatter(sell.index, sell, c='red', label='SELL')
        ax2.set_yticks([])
        ax2.legend(loc=1, fontsize=15)

        plt.xlim(actions.index[0], actions.index[-1])       

        plt.show()
        
    def plot_env(self, states_to_plot=None, start_time=None, end_time=None):
        self.env.plot(states_to_plot, start_time, end_time)
        
    def plot_portfolio(self, states_to_plot=None, start_time=None, end_time=None):
        self.portfolio.plot(states_to_plot, start_time, end_time)