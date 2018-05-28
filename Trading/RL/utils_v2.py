'''
utils_v2
1. Include 3 additional internal states: "starting_cash", "steps_left_in_episode" and "last_buy_price". 
2. Use spread = 0. 
3. Use 'USDT_BTC_5min_mean' from 'df_hourly_BTC_trading_5min_mean.pickle' instead of 'USDT_BTC_open' for transactions. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
class Action(Enum):
    SELL=0
    HOLD=1
    BUY=2

# features
external_state_list = ['USDT_BTC_high', 'USDT_BTC_low', 'USDT_BTC_close', 'USDT_BTC_open', \
           'USDT_BTC_weighted_mean', \
           'USDT_BTC_volatility', 'USDT_BTC_pctChange', 'USDT_BTC_open_label', 'USDT_BTC_pctChange_label', 'USDT_BTC_volatility_label']

# portfolio information 
internal_state_list = ["coin", "cash", "total_value", "is_holding_coin", "return_since_entry", \
                       "starting_cash", "steps_left_in_episode", "last_buy_price"] 

spread = 0.0 / 100 # BTC spread
 
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError ("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


'''
idx : index of the episode in the test_history_list 
choice:  'b' stands for baseline, currently not implemented yet.
'p' stands for partition, whether to show coin and cash split or not
'g' stands for good/bad choice.
'''
def plot_test(agent, test_history_list, idx, choice = 'bp'):
    
    start_time = test_history_list[idx][0].to_datetime()
#     print(start_time) 
    end_time = test_history_list[idx][1].to_datetime()
    test_actions = test_history_list[idx][2]
    test_cash_val = test_history_list[idx][3][0]
    test_coin_val = test_history_list[idx][3][1]
    test_portfolio_val = test_history_list[idx][3][2]

    if end_time is None: # default: one day
        end_time = agent.env.end_index
        
    df = agent.env.df
    df = df.loc[df.index >= start_time]
    df = df.loc[df.index <= end_time]
    prices = df['USDT_BTC_open']
    #prices = df['USDT_BTC_5min_mean']
#     print(prices.shape)
#     print(smooth(prices))

    ts = agent.env.df.ix[start_time:end_time].index
    portfolio_values = pd.Series(test_portfolio_val, index=ts)
    actions = pd.Series(test_actions, index=ts)

    print(len(test_actions))
    print(len(prices))
    print(len(test_cash_val))

#     actions = agent.test_actions
#     actions = actions[actions.index >= start_time]
#     actions = actions[actions.index < end_time]

    fig, ax1 = plt.subplots(figsize = (15, 8))

    ax1.plot(prices.index, prices, 'b-')
    ax1.set_ylabel('Price', color='b', fontsize=15)
    ax1.tick_params('y', colors='b', labelsize=15)

    hold = actions[actions == 1]
    buy = actions[actions == 2]
    sell = actions[actions == 0]
    
#     print(len(hold))
#     print(len(sell))


    if 'g' in choice: 
        sm =  smooth(prices,24)[12:len(prices.index)+12]
        import numpy as np
        from scipy.signal import argrelextrema

        local_minima = argrelextrema(sm,np.less) 
        local_maxima = argrelextrema(sm,np.greater)
        turning = np.concatenate((local_minima[0],local_maxima[0]),axis=0)
        turning = np.append(turning,0)
        turning = np.append(turning,len(prices.index) - 1)
        turning.sort()
        sell_first = True
        if (prices[0] < prices[1]):
            sell_first = False
        l_turning = list(turning)
        edge = []
        edge.append(0)
        for i in range(len(l_turning) - 1):
            edge.append((l_turning[i] + l_turning[i + 1])//2)
        edge.append(len(prices.index) - 1)

        good_action = []
        good_action.append(1)
        cur_action = 0 # sell
        last_edge = 0
        if (not sell_first):
            cur_action = 2 # buy
        for i in range(1,len(edge)):
            target_edge = edge[i]
            for j in np.arange(last_edge,target_edge,1):
                good_action.append(cur_action)
            cur_action = 2 - cur_action
            last_edge = target_edge
#         print(len(good_action))
        good_action_df = pd.DataFrame()
        good_action_df["good_action"] = good_action
        good_action_df.index = actions.index




        good_buy = actions[(actions == 2) & (actions == good_action)]
        bad_buy = actions[(actions== 2) & (actions != good_action)]
        good_sell = actions[(actions == 0) & (actions == good_action)]
        bad_sell = actions[(actions== 0) & (actions != good_action)] 

        ax2 = ax1.twinx()
    #     print(hold)
    #     print(actions)
        if (len(hold) != 0):
            ax2.scatter(hold.index, hold, c='blue', label='HOLD')
        if (len(good_buy) != 0):
            ax2.scatter(good_buy.index, good_buy, c='green', marker = 'o',label='GOOD_BUY')
        if (len(bad_buy) != 0):    
            ax2.scatter(bad_buy.index, bad_buy, c='red',marker = 'x', label='BAD_BUY')
        if (len(good_sell) != 0):
            ax2.scatter(good_sell.index, good_sell, c='green', marker = 'o', label='GOOD_SELL')
        if (len(bad_sell) != 0):
            ax2.scatter(bad_sell.index, bad_sell, c='red', marker = 'x', label='BAD_SELL')
        ax2.set_yticks([])
        ax2.legend(loc=1, fontsize=15)
        
    
#     ax3 = ax1.twinx()
#     ax3.plot(prices.index,sm, 'r-')
#     ax3.tick_params('y_smooth', colors='r', labelsize=15)
#     ax3.set_yticks([])

    else:
        ax2 = ax1.twinx()
        if (len(hold)!=0):
            ax2.scatter(hold.index, hold, c='blue', label='HOLD')
        if (len(buy) != 0):
            ax2.scatter(buy.index, buy, c='green', label='BUY')
        if (len(sell) != 0):
            ax2.scatter(sell.index, sell, c='red', label='SELL')
        ax2.set_yticks([])
        ax2.legend(loc=1, fontsize=15)

    ax4 = ax1.twinx()
#     ax4.set_ylim(0,6000)
    if 'p' in choice:
        ax4.plot(prices.index, test_cash_val, 'y', label='Cash Value')
        ax4.plot(prices.index, test_coin_val, 'orange', label='Coin Value in USD')
    ax4.plot(prices.index, portfolio_values ,'purple',label='Portfolio Value in USD')
#     if 'b' in choice:
#         ax4.plot(prices.index,agent.baseline_portfolio_values,'red',label = 'Baseline Portfolio Value')
    ax4.legend(loc=4, fontsize=15)

    plt.xlim(actions.index[0], actions.index[-1])       

    plt.show()


    
    
class Environment:
    def __init__(self, coin_name="BTC", states=external_state_list, recent_k = 0):
        dir_path = '../../Data/'
        df = pd.read_pickle(dir_path+'df_hourly_BTC_trading_5min_mean.pickle')
        self.df = df.dropna()
        self.length = len(self.df.index)
        self.current_index = self.df.index[0]
        self.start_index = self.df.index[0]
        self.end_index = self.df.index[-1]
        self.time_delta = self.df.index[1] - self.df.index[0]
        self.coin_name = coin_name
        self.states = states
        
    def getStates(self):
        s = list(self.df.loc[self.current_index][self.states])
#         print(s)
        return s
    
    def getStatesSequence(self, seq_len=10):
        ss = self.df.loc[(self.current_index - (seq_len-1) * self.time_delta) : self.current_index][self.states].as_matrix()
#         print(ss)
        ss = ss.reshape([1,-1]).tolist()[0]
        return ss
        
    def getStateSpaceSize(self):
        return len(self.states)
    
    # [start_time, end_time)
    def step(self, end_time=None):
        if self.current_index == end_time:
            return True, self.getStates()
        self.current_index += self.time_delta
        return self.current_index == self.df.index[-1], self.getStates()
        
    def reset(self):
        self.current_index = self.df.index[0]
        
    def set_current_time(self, current_time):
        self.current_index = current_time
        
    def getFinalPrice(self):
        return self.df.iloc[self.length-1]['USDT_BTC_5min_mean'] 
        
    def getCurrentPrice(self):
        return self.df.loc[self.current_index]['USDT_BTC_5min_mean'] 
    
    def getPriceAt(self, index):
        if index < self.df.index[0]:
            return self.df.iloc[0]['USDT_BTC_5min_mean']
        if index >= self.df.index[self.length-1]:
            return self.getFinalPrice()
        return self.df.loc[index]['USDT_BTC_5min_mean'] 
    
    
    def getReward(self, action):
        a = 0
        if action == Action.BUY:
            a = 1
        elif action == Action.SELL:
            a = -1
            
        price_t = self.getCurrentPrice()
        price_t_minus_1 = self.getPriceAt(self.current_index - self.time_delta)
        price_t_minus_n = self.getPriceAt(self.current_index - self.time_delta*10)
        
        r = (1 + a*(price_t - price_t_minus_1)/price_t_minus_1)*price_t_minus_1/price_t_minus_n
        return r
    
    def plot(self, states_to_plot=None, start_time=None, end_time=None):
        import matplotlib.pyplot as plt
        if not states_to_plot:
            states_to_plot = self.states

        plt.figure()
        
        df = self.df
        if start_time:
            df = df.loc[df.index >= start_time]
        if end_time: 
            df = df.loc[df.index <= end_time]
        
        for state in states_to_plot:
            ax = df[state].plot()
        ax.legend(states_to_plot)
        plt.show()

    
class Portfolio:
    def __init__(self, cash_supply, num_coins_per_order=1.0, states=internal_state_list, verbose=False, final_price=0.0):
        self.portfolio_coin = 0.0
        self.portfolio_cash = cash_supply
        self.starting_cash = cash_supply
        self.num_coins_per_order = num_coins_per_order
        self.final_price = final_price
        self.states = states
        self.verbose = verbose
        
        ### Mapping states to their names, do we need this?
        self.state_dict = {}
        self.state_dict["coin"] = self.portfolio_coin
        self.state_dict["cash"] = self.portfolio_cash
        self.state_dict["total_value"] = self.portfolio_cash
        self.state_dict["is_holding_coin"] = 0
        self.state_dict["return_since_entry"] = 0
        self.state_dict["starting_cash"] = self.starting_cash
        self.state_dict["steps_left_in_episode"] = None
        self.state_dict["last_buy_price"] = None
        
        self.bought_price = 0.0
        
    # return internal state    
    def getStates(self, states=None):
        if not states:
            states = self.states
        return [self.state_dict[state] for state in states]
        
    def getStateSpaceSize(self):
        return len(self.states)
    
    def getActionSpaceSize(self):
        return len(list(Action))
    
    # reset portfolio
    def reset(self):
        self.__init__(cash_supply=self.starting_cash, num_coins_per_order=self.num_coins_per_order, 
                      states=self.states, verbose=self.verbose, final_price=self.final_price)
        
    # apply action (buy, sell or hold) to the portfolio
    # update the internal state after the action
    def apply_action(self, current_price, action, verbose, xaction_fee=0.25/100):
        self.state_dict["total_value"] = self.getCurrentValue(current_price)
        if verbose:
            print("Action start:", action, ", Total value before action:", self.state_dict["total_value"])           
        
        if str(action) == 'Action.BUY':
            action = Action.BUY
            coin_to_buy, buy_price = self.__buy(current_price, verbose, xaction_fee)
            if coin_to_buy > 0:
                self.bought_price = buy_price               
            else:
                action = Action.HOLD
                
        elif str(action) == 'Action.SELL':
            action = Action.SELL
            coin_to_sell, sell_price = self.__sell(current_price, verbose, xaction_fee)
            if coin_to_sell > 0:
                pass
                
            else:
                action = Action.HOLD
        
        # Update states
        self.state_dict["coin"] = self.portfolio_coin
        self.state_dict["cash"] = self.portfolio_cash
        self.state_dict["total_value"] = self.getCurrentValue(current_price)
        self.state_dict["is_holding_coin"] = (self.portfolio_coin > 0)*1
        self.state_dict["return_since_entry"] = self.getReturnsPercent(current_price)
        
        if verbose:
            print("Action end: ", action, ", Total value now: %.3f. "%self.state_dict["total_value"],", Return since entry: %.3f %%" %(self.state_dict["return_since_entry"]))
            print()
            
        return action
    
    def __buy(self, current_price, verbose, xaction_fee=0.25/100):
        if not current_price:
            return 0
        
        buy_price = current_price * (1 + spread)     
        
#         coin_to_buy = min(self.num_coins_per_order, np.floor(self.portfolio_cash / current_price))
        coin_to_buy = self.portfolio_cash / buy_price * 1.0 / 10
        
        if verbose:
            print("Before buying: coin:%.3f, cash:%.3f, buy price:%.3f" %(
                self.portfolio_coin, self.portfolio_cash, buy_price))
            
        self.portfolio_coin += coin_to_buy
        self.portfolio_cash -= coin_to_buy * buy_price 
        fees = coin_to_buy * buy_price * xaction_fee # assume 0.25% transaction fees
        self.portfolio_cash -= fees
        
        if verbose:
            print("After buying: coin bought:%.3f, transaction fees:%.3f, coin now:%.3f, cash now:%.3f" %(
                coin_to_buy, fees, self.portfolio_coin, self.portfolio_cash))
        
        return coin_to_buy, buy_price
    
    def __sell(self, current_price, verbose, xaction_fee=0.25/100):
        if not current_price:
            return 0
        
        sell_price = current_price * (1 - spread)    
        
#         coin_to_sell = min(self.num_coins_per_order, self.portfolio_coin)
        coin_to_sell = self.portfolio_coin * 1.0 / 10
        
        if verbose:
            print("Before selling: coin:%.3f, cash:%.3f, sell price:%.3f" %(
                self.portfolio_coin, self.portfolio_cash, sell_price))
        
        self.portfolio_coin -= coin_to_sell
        self.portfolio_cash += coin_to_sell * sell_price
        fees = coin_to_sell * sell_price * xaction_fee # assume 0.25% transaction fees
        self.portfolio_cash -= fees
        
        if verbose:
            print("After selling: coin sold:%.3f, transaction fees:%.3f, coin now:%.3f, cash now:%.3f" %(
                coin_to_sell, fees, self.portfolio_coin, self.portfolio_cash))
        
        return coin_to_sell, sell_price
    
    def getCurrentValue(self, current_price):
        sell_price = current_price * (1 - spread)
        return self.portfolio_coin * sell_price + self.portfolio_cash
        
    def getReturnsPercent(self, current_price):
        return 100 * (self.getCurrentValue(current_price) - self.starting_cash) / self.starting_cash

    def getCurrentHoldings(self, current_price):
        return "%.2f coins, %.2f cash, %.2f current value, %.2f percent returns" \
                    % (self.portfolio_coin, self.portfolio_cash, \
                       self.getCurrentValue(current_price),self.getReturnsPercent(current_price))
