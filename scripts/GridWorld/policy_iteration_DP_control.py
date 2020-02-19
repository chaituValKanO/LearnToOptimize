from __future__ import print_function, division
from builtins import range

from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_DP_prediction import print_policy, print_values
import numpy as np

SMALL_VALUE = 1e-3
GAMMA = 0.9

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == "__main__":
    grid = negative_grid()
    print("Rewards:")
    print_values(grid.rewards, grid)

    # Initialize policy with random actions
    policy = {}
    for state in grid.actions.keys():
        policy[state] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    print("Initial Policy")
    print_policy(policy, grid)

    #Initialize V
    V = {}
    for state in grid.all_states():
        if state in grid.actions:
            V[state] = np.random.random()
        else:
            V[state] = 0
    
    print("Initial Value function")
    print_values(V, grid)

    while True:
        while True:
            biggest_change = 0
            for state in grid.all_states():
                old_v = V[state]
                if state in policy:
                    grid.set_state(state)
                    r = grid.move(policy[state])
                    V[state] = r + GAMMA * V[grid.current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v - V[state]))
            
            if biggest_change < SMALL_VALUE:
                break
        
        # Policy Improvement Step

        is_policy_converged = True

        for state in grid.all_states():
            if state in policy:
                old_a = policy[state]
                new_a = None
                best_value = float('-inf')
                #for the above new value function we have to update the policy to find best action for the state
                #among the availble actions
                for action in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(state)
                    r = grid.move(action)
                    v = r + GAMMA * V[grid.current_state()]

                    if v > best_value:
                        best_value = v
                        new_a = action
                policy[state] = new_a
                if new_a != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break
    
    print("Updated Policy")
    print_policy(policy, grid)

    print("updated value function")
    print_values(V, grid)
                






        


