'''
Mint Agent, ycli
[Requires utils_v2.py]
Based on Vanilla DQ Agent. Changes include:
1. Change neural network to 2 layers to overcome vanishing gradients. Also changed stddev in initialization.
2. Require 'cash_supply' for __init__()
3. Include 'seq_len' as an argument for __init__(). 'seq_len' is the number of hours the agent sees before making a decision. Defaults to 10.
4. Change self.batch_size from 32 to 1800.
5. Include 'USDT_BTC_open_label', 'USDT_BTC_pctChange_label', 'USDT_BTC_volatility_label' as state features. See utils_v2.py.
6. Change self.env.getStates() to self.env.getStatesSequence(), which is a new method implemented in utils_v2.py
7. Implement self.state_mean in __replay() for feature centering. Not sure if this is necessary. [problem: this is not saved with the model, so when you load the model later you lose this state_mean.]
8. In __replay(), roll the reward column of self.memory for 9 hours. Note that this does not need to match seq_len.
9. In train(), change reward definition to the percent change of portfolio value. This, with the rolling mentioned above, makes the reward the relative change in portfolio value of the 10th hour in the future with respect to the 9th hour in the future. (This is still   not a natural reward function, right?)
10. In test(), the model is not automatically loaded. Instead, load the model explicitly in your ipynb.
11. In train() and test(), change the order of env.step() and portfolio.apply_action(). Step first.
12. Changed keras implementation to tensorflow.
13. Changed activation to leaky_relu !!!!!
'''

from importlib import reload
import utils_v2
reload(utils_v2)
from utils_v2 import *

import tensorflow as tf

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from keras import backend as K
# from keras import initializers
# from keras.models import load_model

# Neural Network for the Q value approximation
class QValue_NN:
    def __init__(self, state_size, action_size, units):
        self._state_size = state_size
        self._action_size = action_size
        self._units = units
        self.__build_model()
        
    def __huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def __build_model(self): # maybe fix random seed here
        
            self.X = tf.placeholder(tf.float32, [None, self._state_size])
            self.y = tf.placeholder(tf.float32, [None, self._action_size])
            a1 = tf.layers.dense(self.X, self._action_size, activation=tf.nn.leaky_relu)  # first FC
            self.preds = tf.layers.dense(a1, self._action_size, activation=tf.nn.leaky_relu)  # second FC
            self.loss = tf.nn.l2_loss(self.preds - self.y)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01) 
            self.train_op = optimizer.minimize(self.loss)

    def train(self, session, state, qvalues):
        state_reshape = np.reshape(state, [1, len(state)])
        session.run(self.train_op, feed_dict={self.X: state_reshape, self.y:qvalues})

    def predict(self, session, state):
        state_reshape = np.reshape(state, [1, len(state)])
        return session.run(self.preds, feed_dict={self.X: state_reshape})
    
#     def set_weights(self, model_weights):
#         self._model.set_weights(model_weights)
        
#     def get_weights(self):
#         return self._model.get_weights()
    
#     def save(self, path):
#         self._model.save_weights(path)
        
#     def load(self, path):
#         self._model.load_weights(path)
        

import random
import numpy as np
from collections import deque
import pandas as pd

# Agent Implementation
class MintDQNAgent:
    
    # initialize internal variables
    def __init__(self, cash_supply, input_seq_len=10, gamma=0.95, num_neutron=24, epsilon_min = 0.001, epsilon_decay=0.995, 
                 coin_name='BTC', num_coins_per_order=1.0, recent_k = 0,
                 external_states = external_state_list,
                 internal_states = internal_state_list, verbose=False):
        self.max_mem_len = 2000
        self.memory = deque(maxlen=2000)
        self.batch_size = 1800
        self.gamma = gamma
        self.epsilon=1.0
        self.epsilon_min=epsilon_min 
        self.epsilon_decay=epsilon_decay
        self.coin_name = coin_name
        # External states
        self.external_states = external_states
        self.env = Environment(coin_name=coin_name, states=external_states, recent_k=recent_k)
        # Internal states
        self.internal_states = internal_states
        self.portfolio = Portfolio(cash_supply=cash_supply, num_coins_per_order=num_coins_per_order, states=internal_states,
                                   verbose=verbose, final_price=self.env.getFinalPrice())
        # NN model
        _state_size = self.env.getStateSpaceSize()*input_seq_len + self.portfolio.getStateSpaceSize()
        tf.reset_default_graph()
        with tf.variable_scope("model", initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.model = QValue_NN(_state_size, self.portfolio.getActionSpaceSize(), num_neutron)
            self.old_vars = {v.name.split('model/')[-1] : v for v in tf.trainable_variables() if v.name.startswith(scope.name + "/")}
        with tf.variable_scope("target_model", initializer=tf.contrib.layers.xavier_initializer()):
            self.target_model = QValue_NN(_state_size, self.portfolio.getActionSpaceSize(), num_neutron)
        
        self.train_cum_returns = []
        
        self.test_cum_returns = []
        self.test_portfolio_values = []
        self.test_actions = []
        self.seq_len = input_seq_len
        
        self.state_mean = None
     
    def plot_external_states(self):
        self.env.plot(self.external_states)
    
    
    def __act(self, session, state):
        if np.random.rand() < self.epsilon:
            return random.choice(list(Action))
        act_values = self.model.predict(session, state)
        print(act_values)
#         if np.array_equal(act_values, np.array([[-1.0, -1.0, -1.0]])):
#             print("what???????????????????????????????????????")
#             print(state)
        return Action(np.argmax(act_values[0]))
        
    def __remember(self, state, action, reward, next_state, isDone):
        self.memory.append((state, action, reward, next_state, isDone))
        
    def __update_target_model(self, sess):
        #         self.target_model._model.set_weights(self.model._model.get_weights())
        with tf.variable_scope("target_model", initializer=tf.contrib.layers.xavier_initializer()) as scope:
#             for v in tf.trainable_variables():
#                 print(v.name)
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
        
    def __replay(self, session, batch_size):
        # key: some delay here
            
#         print(self.memory[:,0])
#         print('mean state:', np.mean(self.memory[:,0]))
        
        self.memory = np.array(self.memory)
        self.memory[:,2] = np.roll(self.memory[:,2], -9, axis=0)
        self.memory = deque(self.memory[:-9,:], maxlen=self.max_mem_len)
        print(len(self.memory))
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, isDone in minibatch:
#             state -= self.state_mean
#             next_state -= self.state_mean
#             print('state',state)
            target = self.model.predict(session, state)
#             print('target predict before action:', target)
            if isDone:
                target[0][action.value] = reward
            else:
                a = self.model.predict(session, next_state)[0]
                t = self.target_model.predict(session, next_state)[0]
                
                # Bellman Equation
                target[0][action.value] = reward + self.gamma * t[np.argmax(a)]
#                 print('action:',action)
#                 print('target predict after action:', target)
#                 print('======')

            self.model.train(session, state, target)
        
        # update the epsilon to gradually reduce the random exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Agent Training
    
    ### Sample Usage:
        
    ### import datetime
    ### end = datetime.datetime(2018,1,1,0)
    ### agent.train(end_time = end)
    
    def train(self, session, end_time, num_episodes=100, start_time=None, verbose=True):
        session.run(tf.global_variables_initializer())
        self.cum_returns = []
        
        if start_time is None:
                start_time = self.env.start_index
        
        n_days = (end_time - start_time) // (self.env.time_delta * 24)
        print('Training from ', start_time, ' to', end_time, ': ', '~', n_days, 'days\n')
        
        for i in range(num_episodes):
            
            self.env.reset()
            self.portfolio.reset()
            self.env.set_current_time(start_time)
            state = self.env.getStatesSequence() + self.portfolio.getStates()

            # walk through the environment
            # obtain action based on state values using the Neural Network model
            # collect reward
            # update the experience in Memory
            while (True):
                if verbose:
                    print('Current time:', self.env.current_index)
                    
                value_before_action = self.portfolio.getCurrentValue(self.env.getCurrentPrice())
                
                if self.state_mean is not None:
                    action = self.__act(session, state - self.state_mean)
                else:
                    action = self.__act(session, state)
                isDone, next_state = self.env.step(end_time) # order changed
                action = self.portfolio.apply_action(self.env.getCurrentPrice(), action, verbose)
                
                next_state = self.env.getStatesSequence() # mint
                next_state = next_state + self.portfolio.getStates()
                
                if self.state_mean is not None:
                    next_state -= self.state_mean
                    
#                 reward = self.env.getReward(action) # this was used in vanilla
                reward = (self.portfolio.getCurrentValue(self.env.getCurrentPrice()) / value_before_action - 1) * 100
                if verbose:
                    print(action, reward)
                    print()
                
                self.__remember(state, action, reward, next_state, isDone)
                state = next_state
                
                if i == 0:
                    # estimate state mean
                    memory = np.array(self.memory)
                    state_sum = 0
                    for list_of_state in memory[:,0]:
                        state_sum += np.array(list_of_state)
                    self.state_mean = state_sum / len(memory)
                
                if isDone:
                    self.__update_target_model(session)
                    
                    cum_return = self.portfolio.getReturnsPercent(self.env.getCurrentPrice())
                    self.train_cum_returns.append(cum_return)
                    
                    print("episode: {}/{}, returns: {:.2}, epsilon: {:.2}"
                          .format(i+1, num_episodes, 
                                  cum_return, 
                                  self.epsilon))
                    break
             
            # train the Neural Network incrementally with the new experiences
            if len(self.memory) > self.batch_size:
                self.__replay(session, self.batch_size)
                
        self.target_model.save('{}.model.h5'.format(self.coin_name))
                
        
    ### Sample Usage:
        
    ### import datetime
    ### start = datetime.datetime(2018,1,1,0)
    ### agent.test(start_time = start)
    
    def test(self, session, start_time, end_time=None, epsilon=None, verbose=True, print_freq='daily'):
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = 0 # set to 0, no randomness allowed 
        
        self.env.reset()
        self.env.set_current_time(start_time)
        self.portfolio.reset()
        state = self.env.getStatesSequence() + self.portfolio.getStates()
        state -= self.state_mean
#         self.model.load('{}.model.h5'.format(self.coin_name))
        
        self.test_cum_returns = []
        self.test_portfolio_values = []
        self.test_actions = []
        
        if end_time is None or end_time >= self.env.end_index:
            end_time = self.env.end_index - self.env.time_delta
        
        n_days = (end_time - start_time) // (self.env.time_delta * 24)
        print('Testing from ', start_time, ' to', end_time, ': ', '~', n_days, 'days\n')
    
        start_day = start_time.day
        verbose_g = verbose
    
        while (True):
            
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
                    if self.env.current_index.day in np.roll((np.arange(28)+1), 28-start_day+1)[::7] and self.env.current_index.hour == 0:
                        print('Current time:', self.env.current_index)
                        verbose = True
                    else:
                        verbose = False
            
            action = self.__act(session, state)
            isDone, next_state = self.env.step(end_time) # order changed
            action = self.portfolio.apply_action(self.env.getCurrentPrice(), action, verbose)
            
            next_state = self.env.getStatesSequence()
            next_state = next_state + self.portfolio.getStates()
            state = next_state
            state -= self.state_mean
            
            cum_return = self.portfolio.getReturnsPercent(self.env.getCurrentPrice())
            self.test_cum_returns.append(cum_return)
                
            portfolio_value = self.portfolio.getCurrentValue(self.env.getCurrentPrice())
            self.test_portfolio_values.append(portfolio_value)
            
            self.test_actions.append(action.value)
            
            verbose = verbose_g
            
            if isDone:
                break
        
        ts = self.env.df.ix[start_time:end_time].index
        self.test_cum_returns = pd.Series(self.test_cum_returns, index=ts)
        self.test_portfolio_values = pd.Series(self.test_portfolio_values, index=ts)
        self.test_actions = pd.Series(self.test_actions, index=ts)

        print('Percentage return:', self.portfolio.getReturnsPercent(self.env.getCurrentPrice()))
        
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