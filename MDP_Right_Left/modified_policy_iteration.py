'''
Modified Policy Iteration
'''
def modified_policy_evaluation (
    reward: list, transitional_probability: list, discount_factor: float, value: list, \
        k: int, terminal: list, state_num: int, action_num: int):

    delta_list = []
    tp = transitional_probability

    # iterate with k times
    for i in range(k):
        delta = 0
        NewValue = [-1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12]
        print(str(i) + ': ', \
            f'{value[0]:.2f}', f'{value[1]:.2f}', f'{value[2]:.2f}', f'{value[3]:.2f}',\
            f'{value[4]:.2f}', f'{value[5]:.2f}', f'{value[6]:.2f}', f'{value[7]:.2f}',\
            'bellman_factor (' + str(round(delta, 3)) + ')', sep=",    ")
        for row in range(state_num):
            if terminal[row] == 'T':
                NewValue[row] = reward[row][0]
                continue
            value_temp = 0
            for action in range(action_num):
                for column in range(state_num):
                    value_temp += tp[action][row][column] * value[column] \
                        * discount_factor + reward[row][action]
            NewValue[row] = round(value_temp, 2)
        for i in range(state_num):
            delta = max(delta, abs(value[i] - NewValue[i]))
        delta_list.append(round(delta, 2))

        if i <= k:
            value = NewValue
        else:
            return NewValue

def modified_policy_improvement(
    value: list, transitional_probability: list, discount_factor: float, \
        given_policy: list, terminal: list, state_num: int):    
    
    CHANGE = False
    RIGHT = 0
    LEFT = 1
    tp = transitional_probability
    right_value = 0
    left_value=  0
    new_policy = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
       
    # for each states, determine "right value" or "left value" is better
    for row in range(state_num): 
        for column in range(state_num):
        # right value
            right_value += tp[RIGHT][row][column] * value[column]
            right_value *= discount_factor
        # left value
            left_value += tp[LEFT][row][column] * value[column]
            left_value *= discount_factor
        if terminal[row] != 'T':
            if right_value > left_value:
                new_policy[row] = 'r'
            else:
                new_policy[row] = 'l'
        else:
            new_policy[row] = 'T'

        # if any state's policy different from old policy, then return True
        if new_policy[row] != given_policy[row]: 
            CHANGE = True
    return new_policy, CHANGE

def modified_policy_iteration(
    reward: list, initial_value: list, transitional_probability: list, discount_factor: float, \
        initial_policy: list, initial_terminal: list, k: int, state_num: int, action_num: int):

    STABLE = False
    delta_list_all = []

    while not STABLE:
        converge_value= modified_policy_evaluation(reward, transitional_probability, \
            discount_factor, initial_value, k, initial_terminal, state_num, action_num)
        new_policy, CHANGE = modified_policy_improvement(converge_value, transitional_probability, \
            discount_factor, initial_policy, initial_terminal, state_num)

        if CHANGE == False:
            STABLE = True
        else:
            initial_value = converge_value
            initial_policy = new_policy

    return new_policy

