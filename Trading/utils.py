import pandas as pd
import numpy as np

state_list = ["coin", "cash", "total_value", "is_holding_coin", "return_since_entry"] 

# spread = 0.68 / 100 # BTC spread
spread = 0 # should fix this later

from enum import Enum
class Action(Enum):
    HOLD=0
    BUY=1
    SELL=2
    
class Environment:
    def __init__(self):
        dir_path = '../CS341-repo/Data/'
        df = pd.read_pickle(dir_path+'df_hourly_BTC_with_labels.pickle')
        self.df = df.dropna()
        self.current_index = self.df.index[0]
        self.start_index = self.df.index[0]
        self.time_delta = self.df.index[1] - self.df.index[0]
        
    def getState(self):
        return self.df.loc[self.current_index,['USDT_BTC_high', 'USDT_BTC_low', 'USDT_BTC_close', 'USDT_BTC_open', \
           'USDT_BTC_volume', 'USDT_BTC_quoteVolume', 'USDT_BTC_weighted_mean', \
           'USDT_BTC_volatility', 'USDT_BTC_pctChange']]
    
    def step(self):
        self.current_index += self.time_delta
        return self.current_index == self.df.index[-1], self.getState()
        
    def reset(self):
        self.current_index = self.df.index[0]
        
    def set_current_time(self, current_time):
        self.current_index = current_time
        
    def getCurrentPrice(self):
        return self.df.loc[self.current_index, 'USDT_BTC_open'] # is this the correct 'current price'??

    
class PredictEnvironment:
    def __init__(self):
        dir_path = '../Data/'
        df = pd.read_pickle(dir_path+'df_hourly_BTC_with_labels.pickle')
        predict_df = pd.read_pickle(dir_path+'df_hourly_BTC_with_6_test_prediction.pickle')
        self.df = df.dropna()
        #  start from test time.
        self.predict_df = predict_df.dropna()
        self.current_index = self.df.index[0]
        self.start_index = self.df.index[0]
        self.time_delta = self.df.index[1] - self.df.index[0]


    def getState(self):
        return self.df.loc[self.current_index,['USDT_BTC_high', 'USDT_BTC_low', 'USDT_BTC_close', 'USDT_BTC_open', \
           'USDT_BTC_volume', 'USDT_BTC_quoteVolume', 'USDT_BTC_weighted_mean', \
           'USDT_BTC_volatility', 'USDT_BTC_pctChange']]

    def getPredict(self):
        next_index = self.current_index + self.time_delta
        return self.predict_df.loc[next_index,['open_pctChange_predict','close_pctChange_predict',\
        'high_pctChange_predict','low_pctChange_predict','mean_pctChange_predict',\
        'volatility_predict']]

    def step(self):
        self.current_index += self.time_delta
        return self.current_index == self.df.index[-2], self.getState(), self.getPredict()
        
    def reset(self):
        self.current_index = self.df.index[0]
        
    def set_current_time(self, current_time):
        self.current_index = current_time
        
    def getCurrentPrice(self):
        return self.df.loc[self.current_index, 'USDT_BTC_open'] # is this the correct 'current price'??

    
class Portfolio:
    def __init__(self, cash_supply, cash_limit_per_order = None):
        self.portfolio_coin = 0.0
        self.portfolio_cash = cash_supply
        self.starting_cash = cash_supply
        self.cash_limit_per_order = cash_limit_per_order
        self.states = state_list
        
        ### Mapping states to their names, do we need this?
        self.state_dict = {}
        self.state_dict["coin"] = self.portfolio_coin
        self.state_dict["cash"] = self.portfolio_cash
        self.state_dict["total_value"] = self.portfolio_cash
        self.state_dict["is_holding_coin"] = 0
        self.state_dict["return_since_entry"] = 0
        
        self.bought_price = 0.0
        
#         self.cash_used = 0.0
        
    # apply action (buy, sell or hold) to the portfolio
    # update the internal state after the action
    def apply_action(self, current_price, action, verbose):
        self.state_dict["total_value"] = self.getCurrentValue(current_price)
        if verbose:
            print("Action start", action, "Total value before action", self.state_dict["total_value"])           
        
#         self.reward = self.getCurrentValue(self.final_price) - self.state_dict["total_value"] # Reward for HOLD
        if action == Action.BUY:
            coin_to_buy, buy_price = self.__buy(current_price, verbose)
            if coin_to_buy > 0:
                self.bought_price = buy_price
#                 self.reward = self.getCurrentValue(self.final_price)-self.state_dict["total_value"] - \
#                     spread * current_price * coin_to_buy - self.cash_used # Reward for BUY
            else:
                action = Action.HOLD
                
        elif action == Action.SELL:
            coin_to_sell, sell_price = self.__sell(current_price, verbose)
            if coin_to_sell > 0:
                pass
                #self.reward = (sell_price - self.bought_price) * coin_to_sell # Reward for SELL
#                 self.reward = self.state_dict["total_value"] - self.cash_used
            else:
                action = Action.HOLD
        
        # Update states
        self.state_dict["coin"] = self.portfolio_coin
        self.state_dict["cash"] = self.portfolio_cash
        self.state_dict["total_value"] = self.getCurrentValue(current_price)
        self.state_dict["is_holding_coin"] = (self.portfolio_coin > 0)*1
        self.state_dict["return_since_entry"] = self.getReturnsPercent(current_price)
        
        if verbose:
            print("Action end:", action, ", Total value now: %.3f. "%self.state_dict["total_value"],"Return since entry: %.3f %%" %(self.state_dict["return_since_entry"]))
            print()
            
        return action
    
    def __buy(self, current_price, verbose):
        if not current_price:
            return 0
        
        buy_price = current_price * (1 + spread)     # ??????????
        
        if self.cash_limit_per_order is None:
            coin_to_buy = self.portfolio_cash / buy_price
        else:
            coin_to_buy = min(self.cash_limit_per_order, self.portfolio_cash) / buy_price
        
        if verbose:
            print("Before buying: coin:%.3f, cash:%.3f, buy price:%.3f" %(
                self.portfolio_coin, self.portfolio_cash, buy_price))
            
        self.portfolio_coin += coin_to_buy
        self.portfolio_cash -= coin_to_buy * buy_price
#         self.cash_used += coin_to_buy * buy_price
        
        if verbose:
            print("After buying: coin bought:%.3f, coin now:%.3f, cash now:%.3f" %(
                coin_to_buy, self.portfolio_coin, self.portfolio_cash))
        
        return coin_to_buy, buy_price
    
    def __sell(self, current_price, verbose):
        if not current_price:
            return 0
        
        sell_price = current_price * (1 - spread)    # ??????????
        
        if self.cash_limit_per_order is None:
            coin_to_sell = self.portfolio_coin
        else:
            coin_to_sell = min(self.cash_limit_per_order / sell_price, self.portfolio_coin)
        
        if verbose:
            print("Before selling: coin:%.3f, cash:%.3f, sell price:%.3f" %(
                self.portfolio_coin, self.portfolio_cash, sell_price))
        
        self.portfolio_coin -= coin_to_sell
        self.portfolio_cash += coin_to_sell * sell_price
        
        if verbose:
            print("After selling: coin sold:%.3f, coin now:%.3f, cash now:%.3f" %(
                coin_to_sell, self.portfolio_coin, self.portfolio_cash))
        
        return coin_to_sell, sell_price
    
    def getCurrentValue(self, current_price):
        sell_price = current_price * (1 - spread)  # ??????????
        return self.portfolio_coin * sell_price + self.portfolio_cash
        
    def getReturnsPercent(self, current_price):
#         if self.cash_used == 0.0:
#             return 0.0
#         return 100 * (self.getCurrentValue(current_price) - self.cash_used) / self.cash_used
        return 100 * (self.getCurrentValue(current_price) - self.starting_cash) / self.starting_cash

    def getCurrentHoldings(self, current_price):
        return "%.2f coins, %.2f cash, %.2f current value, %.2f percent returns" \
                    % (self.portfolio_coin, self.portfolio_cash, \
                       self.getCurrentValue(current_price),self.getReturnsPercent(current_price))