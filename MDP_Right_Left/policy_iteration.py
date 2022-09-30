
'''
Policy Iteration
'''
def policy_evaluation (
    reward: list, value: list, transitional_probability: list, discount_factor: float, \
        converge_threshold: float, terminal: list, state_num: int, action_num: int):

    CONVERGE = False
    delta = 0
    iteration = 0
    episode_num = 0
    total_value = []
    tp = transitional_probability
    
    
    while not CONVERGE:
        NewValue = [-1e12 for i in range(state_num)]
        print(str(iteration) + ': ', \
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
                    value_temp += tp[action][row][column] * value[column]
                value_temp *= discount_factor
                value_temp += reward[row][action]
            NewValue[row] = round(value_temp, 2)
        delta = 0
        sub_value = 0
        for i in range(state_num):
            delta = max(delta, abs(value[i] - NewValue[i]))
            sub_value += round(NewValue[i], 3)
        total_value.append(round(sub_value, 3))
        
        episode_num += 1
        if delta < converge_threshold:
            CONVERGE = True
            return NewValue, total_value, episode_num
        else:
            value = NewValue
            iteration += 1
  

def policy_improvement(
    value: list, transitional_probability: list, discount_factor: float, \
    given_policy: list, terminal: list, state_num: int):

    CHANGE = False
    RIGHT = 0
    LEFT = 1
    tp = transitional_probability
    right_value = 0
    left_value=  0
    new_policy = ['NA' for i in range(state_num)]

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

def policy_iteration(
    reward: list, initial_value: list, transitional_probability: list, discount_factor: float,
    converge_threhold: float, initial_policy: list, initial_terminal: list, state_num: int, action_num: int):

    STABLE = False
    total_value = []
    total_episode = 0

    while not STABLE:
        converge_value, sub_value, episode = policy_evaluation(reward, initial_value, transitional_probability, \
            discount_factor, converge_threhold, initial_terminal, state_num, action_num)
        new_policy, CHANGE = policy_improvement(converge_value, transitional_probability, discount_factor, \
            initial_policy, initial_terminal, state_num)

        total_episode += episode
        total_value.extend(sub_value)
        if CHANGE == False:
            STABLE = True
        else:
            initial_value = converge_value
            initial_policy = new_policy

    return new_policy, total_value, total_episode
