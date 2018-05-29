import sys

if len(sys.argv) != 4:
    print('Example usage: python RunCherry.py long_deque CherryLongDeque train')
    print('Example usage: python Runcherry.py reward_priority CherryReward test')
    print('Example usage: python Runcherry.py abs_reward_priority CherryAbsReward cheating')
    quit()

import tensorflow as tf
from utils_v3 import *

from CherryDQNAgent import CherryDQNAgent
import datetime

memory_system = sys.argv[1]
if memory_system not in ['long_deque', 'reward_priority', 'abs_reward_priority']:
    print('unsupported memory_system')
    quit()

agent = CherryDQNAgent(initial_total_value=5000, memory_system=memory_system)

experiment_name = sys.argv[2]

mode = sys.argv[3]

train_start = datetime.datetime(2017,1,1,0)
train_end = datetime.datetime(2018,2,1,0)

sess = tf.Session()
print('memory system is:' + memory_system)
print('experiment_name=', experiment_name)

if mode == 'train':
    agent.train(experiment_name=experiment_name, session=sess, start_time = train_start, \
            end_time = train_end, num_episodes=1000000, episode_len=20, \
            verbose=False, auto_save_and_load=True, save_every=50, test_every=50, reward_func='Andy') 

elif mode == 'cheating':
    test_start = datetime.datetime(2017,1,1,0)
    test_end = datetime.datetime(2018,2,1,0)
    history_list = agent.test(sess, start_time = test_start, end_time = test_end, verbose=False)
    print('Cheating finished.')
    
elif mode == 'test':
    test_start = datetime.datetime(2018,2,1,0)
    test_end = datetime.datetime(2018,4,1,0)
    history_list = agent.test(sess, start_time = test_start, end_time = test_end, verbose=False)
    print('Test finished.')

else:
    print('mode must be train, cheating or test.')