import math
import matplotlib.pyplot as plt
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

Value = [0 for i in range(STATES)]  # Value estimation of each state
Policy = ['NA' for i in range(STATES)]
Terminal = ['', '', '', 'T', '', '', 'T', 'T']

# In case that reward can be different to be in one state with
# different actions we candefine them seperately
Reward = [[0, 0, 0], [2, 2, 2], [1, 1, 1], [-1, -1, -1], \
    [3, 3, 3], [-3, -3, -3], [-7, -7, -7], [5, 5, 5]]

'''
main
'''
# Value Iteration
Converge_Value, ValueIter_Total_Value, ValueIter_Episode = value_evaluation(
    Reward, Value, TransitionProbaility, GAMMA, DELTA, STATES, ACTIONS)
Value_Final_Policy = policy_determination(Reward, Converge_Value, TransitionProbaility, \
    GAMMA, Policy, Terminal, STATES, ACTIONS, ACTION_LIST)
print("Value Iteration algoirthm's final policy is:", Value_Final_Policy)
print(f"Total Value: {ValueIter_Total_Value}")
print(f"Episode: {ValueIter_Episode}")
print("--------------------------------------------------------------------------------------------------------")

# Policy Iteration
Value = [0 for i in range(STATES)]  # Reset Value estimation of each state
Policy = ['NA' for i in range(STATES)] # # Reset Policy estimation of each state
Policy_Final_Policy, PolicyIter_Total_Value, PolicyIter_Episode = policy_iteration(
    Reward, Value, TransitionProbaility, GAMMA, DELTA, Policy, Terminal, STATES, ACTIONS)
print("Policy Iteration algoirthm's final policy is:", Policy_Final_Policy)
print(f"Total Value: {PolicyIter_Total_Value}")
print(f"Episode: {PolicyIter_Episode}")
print("--------------------------------------------------------------------------------------------------------")

# Modified Policy Iteration
Value = [0 for i in range(STATES)]  # Reset Value estimation of each state
Policy = ['NA' for i in range(STATES)] # Reset Policy estimation of each state
Modified_Final_Policy, ModPolicyIter_Total_Value, ModPolicyIter_Episode = modified_policy_iteration(
    Reward, Value, TransitionProbaility, GAMMA, Policy, Terminal, K, STATES, ACTIONS)
print("Modified Policy Iteration algoirthm's final policy is:", Modified_Final_Policy)
print(f"Total Value: {ModPolicyIter_Total_Value}")
print(f"Episode: {ModPolicyIter_Episode}")
print("--------------------------------------------------------------------------------------------------------")

# Plot Converge Speed Result

# extend list length to required length
def extend_list(given_list: list, required_length: int, append_value: float):
    for i in range(required_length):
        if i > len(given_list):
            given_list.append(round(append_value, 3))
    return given_list

# # y value annotation
# def y_value_annotation(x:list, y:list):
#     for x, y, in zip(x, y):
#         plt.annotate(text=str(y), xy=(x, y))


x_axis_length = math.ceil(max(ValueIter_Episode, PolicyIter_Episode, ModPolicyIter_Episode) * 1.2)
x_axis = [i for i in range(x_axis_length -1)]

ValueIter_Total_Value = extend_list(ValueIter_Total_Value, 
                                    x_axis_length, ValueIter_Total_Value[-1])
PolicyIter_Total_Value = extend_list(PolicyIter_Total_Value, 
                                    x_axis_length, PolicyIter_Total_Value[-1])
ModPolicyIter_Total_Value = extend_list(ModPolicyIter_Total_Value, 
                                    x_axis_length, ModPolicyIter_Total_Value[-1])                                    

max_total_value = max(ValueIter_Total_Value[-1], 
                        PolicyIter_Total_Value[-1], 
                        ModPolicyIter_Total_Value[-1])

plt.plot(x_axis, ValueIter_Total_Value, label = "Value Iteration")
plt.plot(x_axis, PolicyIter_Total_Value, label = "Policy Iteration")
plt.plot(x_axis, ModPolicyIter_Total_Value, label = "Modified Policy Iteration")
# y_value_annotation(x_axis, ValueIter_Total_Value)
# y_value_annotation(x_axis, PolicyIter_Total_Value)
# y_value_annotation(x_axis, ModPolicyIter_Total_Value)

plt.xlabel('Episodes')
plt.ylabel('Sum of Values')
plt.title('Markov Decision Process (Right/Left/Back)')
plt.legend()
plt.show()
