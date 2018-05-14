import pandas as pd
import numpy as np

# features
external_state_list = ['USDT_BTC_high', 'USDT_BTC_low', 'USDT_BTC_close', 'USDT_BTC_open', \
           'USDT_BTC_volume', 'USDT_BTC_quoteVolume', 'USDT_BTC_weighted_mean', \
           'USDT_BTC_volatility', 'USDT_BTC_pctChange']

# portfolio information 
internal_state_list = ["coin", "cash", "total_value", "is_holding_coin", "return_since_entry"] 

spread = 0.68 / 100 # BTC spread

from enum import Enum
class Action(Enum):
    SELL=0
    HOLD=1
    BUY=2
    
class Environment:
    def __init__(self, coin_name="BTC", states=external_state_list, recent_k = 0):
        dir_path = '../data/'
        df = pd.read_pickle(dir_path+'df_hourly_BTC_with_labels.pickle')
        self.df = df.dropna()
        self.length = len(self.df.index)
        self.current_index = self.df.index[0]
        self.start_index = self.df.index[0]
        self.end_index = self.df.index[-1]
        self.time_delta = self.df.index[1] - self.df.index[0]
        self.coin_name = coin_name
        self.states = states
        
    def getStates(self):
        return list(self.df.loc[self.current_index][self.states])
        
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
        return self.df.iloc[self.length-1]['USDT_BTC_open'] 
        
    def getCurrentPrice(self):
        return self.df.loc[self.current_index]['USDT_BTC_open'] 
    
    def getPriceAt(self, index):
        if index < self.df.index[0]:
            return self.df.iloc[0]['USDT_BTC_open']
        if index >= self.df.index[self.length-1]:
            return self.getFinalPrice()
        return self.df.loc[index]['USDT_BTC_open'] 
    
    
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
    def __init__(self, cash_supply=1e6, num_coins_per_order=1.0, states=internal_state_list, verbose=False, final_price=0.0):
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
        self.__init__(num_coins_per_order=self.num_coins_per_order, 
                      states=self.states, verbose=self.verbose, final_price=self.final_price)
        
    # apply action (buy, sell or hold) to the portfolio
    # update the internal state after the action
    def apply_action(self, current_price, action, verbose):
        self.state_dict["total_value"] = self.getCurrentValue(current_price)
        if verbose:
            print("Action start:", action, ", Total value before action:", self.state_dict["total_value"])           
        
        if str(action) == 'Action.BUY':
            action = Action.BUY
            coin_to_buy, buy_price = self.__buy(current_price, verbose)
            if coin_to_buy > 0:
                self.bought_price = buy_price               
            else:
                action = Action.HOLD
                
        elif str(action) == 'Action.SELL':
            action = Action.SELL
            coin_to_sell, sell_price = self.__sell(current_price, verbose)
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
    
    def __buy(self, current_price, verbose):
        if not current_price:
            return 0
        
        buy_price = current_price * (1 + spread)     
        
        coin_to_buy = min(self.num_coins_per_order, np.floor(self.portfolio_cash / current_price))
        
        if verbose:
            print("Before buying: coin:%.3f, cash:%.3f, buy price:%.3f" %(
                self.portfolio_coin, self.portfolio_cash, buy_price))
            
        self.portfolio_coin += coin_to_buy
        self.portfolio_cash -= coin_to_buy * buy_price 
        xaction_fee = coin_to_buy * buy_price * 0.25/100 # assume 0.25% transaction fees
        self.portfolio_cash -= xaction_fee
        
        if verbose:
            print("After buying: coin bought:%.3f, transaction fees:%.3f, coin now:%.3f, cash now:%.3f" %(
                coin_to_buy, xaction_fee, self.portfolio_coin, self.portfolio_cash))
        
        return coin_to_buy, buy_price
    
    def __sell(self, current_price, verbose):
        if not current_price:
            return 0
        
        sell_price = current_price * (1 - spread)    
        
        coin_to_sell = min(self.num_coins_per_order, self.portfolio_coin)
        
        if verbose:
            print("Before selling: coin:%.3f, cash:%.3f, sell price:%.3f" %(
                self.portfolio_coin, self.portfolio_cash, sell_price))
        
        self.portfolio_coin -= coin_to_sell
        self.portfolio_cash += coin_to_sell * sell_price
        xaction_fee = coin_to_sell * sell_price * 0.25/100 # assume 0.25% transaction fees
        self.portfolio_cash -= xaction_fee
        
        if verbose:
            print("After selling: coin sold:%.3f, transaction fees:%.3f, coin now:%.3f, cash now:%.3f" %(
                coin_to_sell, xaction_fee, self.portfolio_coin, self.portfolio_cash))
        
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
