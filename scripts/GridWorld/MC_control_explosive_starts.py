from __future__ import print_function, division
from builtins import range

from iterative_policy_evaluation_DP_prediction import print_policy, print_values
from grid_world import standard_grid, negative_grid

import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.9

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def play_game(grid, policy):
    all_states = list(grid.actions.keys())
    start_state_idx = np.random.choice(len(all_states))
    start_state = all_states[start_state_idx]

    grid.set_state(start_state)
    s = grid.current_state()

    a = np.random.choice(ALL_POSSIBLE_ACTIONS)

    state_action_rewards = [(s,a,0)]

    seen_states = set()
    seen_states.add(grid.current_state())

    num_steps = 0

    while True:
        r = grid.move(a)
        s = grid.current_state()
        num_steps += 1

        if s in seen_states:
            reward = -10. /num_steps
            state_action_rewards.append((s, None, reward))
            break
        elif grid.game_over():
            state_action_rewards.append((s,None, r))
            break
        else:
            a = policy[s]
            state_action_rewards.append((s,a,r))
        seen_states.add(s)
    
    state_action_returns = []
    G = 0
    first = True

    for s, a, r in reversed(state_action_rewards):
        if first:
            first =  False
        else:
            state_action_returns.append((s, a, G))
        G = r + GAMMA * G

    state_action_returns.reverse()
    return state_action_returns

def max_dict(dict):
    max_value = float('-inf')
    max_key = None

    for key, value in dict.items():
        if value > max_value:
            max_value = value
            max_key = key
    
    return max_key, max_value

if __name__ == "__main__":
    
    grid = negative_grid(step_cost=-0.9)
    
    print("Rewards:")
    print_values(grid.rewards, grid)

    policy = {}

    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    # initialize Q(s,a) and returns
    Q = {}
    returns = {}
    states = grid.all_states()

    for state in states:
        if state in grid.actions.keys():
            Q[state] = {}
            for action in ALL_POSSIBLE_ACTIONS:
                Q[state][action] = 0
                returns[(state,action)] = []
        else:
            pass
    
    deltas = []
    for t in range(2000):
        if t % 100 ==0:
            print(t)
        
        biggest_change = 0
        seen_state_action_pairs = set()
        states_actions_returns = play_game(grid, policy)
        #print(states_actions_returns)

        for s, a, G in states_actions_returns:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        # update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
    
    plt.plot(deltas)
    plt.show()

    print("final policy:")
    print_policy(policy, grid)

    # find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("final values:")
    print_values(V, grid)