from probability_table_back import TransitionProbaility
from value_iteration import *
from policy_iteration import *
from modified_policy_iteration import *

STATES = 8  # number of states
ACTION_LIST = ['r', 'l', 'b']  # actions
ACTIONS = 3 # number of actions
K = 5 # max iteration number
GAMMA = 0.6
DELTA = 0.01

Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Value estimation of each state
Policy = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
Terminal = ['', '', '', 'T', '', '', 'T', 'T']

# In case that reward can be different to be in one state with
# different actions we candefine them seperately
Reward = [[0, 0, 0], [2, 2, 2], [1, 1, 1], [-1, -1, -1], \
    [3, 3, 3], [-3, -3, -3], [-7, -7, -7], [5, 5, 5]]

'''
main
'''
# Value Iteration
Converge_Value = value_evaluation(Reward, Value, TransitionProbaility, GAMMA, DELTA, STATES, ACTIONS)
Value_Final_Policy = policy_determination(Reward, Converge_Value, TransitionProbaility, \
    GAMMA, Policy, Terminal, STATES, ACTIONS, ACTION_LIST)
print("Value Iteration algoirthm's final policy is:", Value_Final_Policy)
print("--------------------------------------------------------------------------------------------------------")

# Policy Iteration
Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Reset Value estimation of each state
Policy = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'] # # Reset Policy estimation of each state
Policy_Final_Policy = policy_iteration(Reward, Value, TransitionProbaility, GAMMA, \
    DELTA, Policy, Terminal, STATES, ACTIONS)
print("Policy Iteration algoirthm's final policy is:", Policy_Final_Policy)
print("--------------------------------------------------------------------------------------------------------")

# Modified Policy Iteration
Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Reset Value estimation of each state
Policy = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'] # Reset Policy estimation of each state
Modified_Final_Policy = modified_policy_iteration(Reward, Value, TransitionProbaility, GAMMA, \
    Policy, Terminal, K, STATES, ACTIONS)
print("Modified Policy Iteration algoirthm's final policy is:", Modified_Final_Policy)
print("--------------------------------------------------------------------------------------------------------")
