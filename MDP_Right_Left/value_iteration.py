'''
Value Iteration
'''
def value_evaluation (reward: list, value: list, transitional_probability: list, discount_factor: float, \
    converge_threshold: float, state_num: int, action_num: int):

    delta = 0
    episode_num = 0
    max_iteration_num = 100
    total_value = []
    tp = transitional_probability
    

    for iteration in range(max_iteration_num):
        NewValue = [-1e12 for i in range(state_num)]
        print(str(iteration) + ': ', \
            f'{value[0]:.2f}', f'{value[1]:.2f}', f'{value[2]:.2f}', f'{value[3]:.2f}',\
            f'{value[4]:.2f}', f'{value[5]:.2f}', f'{value[6]:.2f}', f'{value[7]:.2f}',\
            'bellman_factor (' + str(round(delta, 3)) + ')', sep=",    ")
        for row in range(state_num):
            for action in range(action_num):
                value_temp = 0
                for column in range(state_num):
                    value_temp += tp[action][row][column] * value[column]
                value_temp *= discount_factor
                value_temp += reward[row][action]
                NewValue[row] = round(max(NewValue[row], value_temp), 2)
        delta = 0
        sub_value = 0
        for i in range(state_num):
            delta = max(delta, abs(value[i] - NewValue[i]))
            sub_value += round(NewValue[i], 3)
        total_value.append(round(sub_value, 3))
        episode_num += 1
        value = NewValue
        
        if(delta < converge_threshold):
            return value, total_value, episode_num

# Determine the policy (One time iteration)
def policy_determination(reward: list, value: list, transition_probability: list, \
    discount_factor: float, policy: list, terminal: list, state_num: int, action_num: int, action_list: list):

    NewValue = [-1e12 for i in range(state_num)]
    for i in range(state_num):
        for a in range(action_num):
            value_temp = 0
            for j in range(state_num):
                value_temp += transition_probability[a][i][j] * value[j]
            value_temp *= discount_factor
            value_temp += reward[i][a]
            if(NewValue[i] < value_temp):
                if(terminal[i] != 'T'):
                    policy[i] = action_list[a]
                    NewValue[i] = max(NewValue[i], value_temp)
                else:
                    policy[i] = 'T'
    return policy
