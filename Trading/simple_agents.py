import random 
# from portfolio import Action

'''
ra = RandomAgent(Action)
print ra.act()
print ra.act()
print ra.act()
'''
class RandomAgent:
    def __init__(self, Action):
        self.Action = Action

    def get_action(self, state=None):
        return random.choice(list(self.Action))

'''
bba = BollingerBandAgent(Action)
print bba.act([0, 0])
print bba.act([1, 0])
print bba.act([0, 1])
'''
# class BollingerBandAgent:

#     def act(self, state):
#         cross_upper_band, cross_lower_band = state 
#         if cross_upper_band:
#             return Action.SELL
#         if cross_lower_band:
#             return Action.BUY
#         return Action.HOLD



class DumbAgent:
	def __init__(self,Action):
		self.Action = Action
	def get_action_by_predict(self,state,next_state):
		if (next_state['mean_pctChange_predict'] >= 4):
			return self.Action.BUY
		if (next_state['mean_pctChange_predict'] <=2):
			return self.Action.SELL
		return self.Action.HOLD