from __future__ import division, print_function
from builtins import range

from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_DP_prediction import print_policy, print_values

import numpy as np

GAMMA = 0.9
SMALL_CHANGE = 1e-3
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R') 

def play_game(grid, policy):
    ''' 
        Function to play an episode. Returns list of state and return [(s,r), ()] 
    '''
    ## Randomly choose the starting point
    all_states = list(grid.actions.keys())
    start_state_idx = np.random.choice(len(all_states))
    grid.set_state(all_states[start_state_idx])

    s = grid.current_state()
    states_and_rewards = [(s,0)]

    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s,r))
    
    G = 0
    states_and_returns = []
    first = True

    for s, r in reversed(states_and_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move

        if first:
            first = False
        else:
            states_and_returns.append((s,G))
        G = r + GAMMA * G
    states_and_returns.reverse()
    return states_and_returns


if __name__ == "__main__":
    grid = standard_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # state -> action
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    ## From play_game function we would get state and corresponding return. As we would be playing 
    ## multiple episodes, we need to seperate state-wise returns and take average of them to get value of that state

    V = {}
    returns = {}

    all_states = grid.all_states()

    for state in all_states:
        if state in grid.actions.keys():
            returns[state] = []
        else:
            V[state] = 0
    
    for rep in range(100):
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns: # Implementation of first visit policy
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)
    
    print("Values:")
    print_values(V, grid)
    print("Policy:")
    print_policy(policy, grid)





