from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_world import standard_grid

SMALL_CHANGE = 1e-3
def print_values(V, g):

    for row in range(g.rows):
        print('-------------------------------')
        for col in range(g.columns):
            v = V.get((row,col), 0)
            if v > 0:
                print(" %.2f |" %v, end="")
            else:
                print("%.2f |" %v, end="")
        print("")

def print_policy(P, g):

    for row in range(g.rows):
        print("---------------------------------")
        for col in range(g.columns):
            a = P.get((row, col), " ")
            print(" %s |" %a, end="")
        print("")

if __name__ == "__main__":
    ### Random policy #####
    g = standard_grid()
    states = g.all_states()

    gamma = 1.0
    V = {}
    for state in states:
        V[state] = 0
    
    while True:
        biggest_change = 0
        for state in states:
            #print(f"State is {state}")
            old_v = V[state]

            if state in g.actions:
                # print(g.actions)
                # print(g.actions[state])
                new_v = 0
                p_a = 1.0/len(g.actions[state])
                for action in g.actions[state]:
                    g.set_state(state)
                    r = g.move(action)
                    new_v += p_a * (r + gamma * V[g.current_state()])
                V[state] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[state]))
        
        if biggest_change < SMALL_CHANGE:
            break
    print("values for uniformly random actions:")
    print_values(V, g)
    print("\n\n")

    ### Fixed Policy ###
    policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U'}

    print_policy(policy, g)

    gamma = 0.9

    for state in states:
        V[state] = 0
    while True:
        biggest_change = 0
        for state in states:
            old_v = V[state]

            if state in policy:
                action = policy[state]
                g.set_state(state)
                r = g.move(action)
                new_v = r + gamma * V[g.current_state()]
                V[state] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[state]))
        
        if biggest_change < SMALL_CHANGE:
            break
    
    print("values for fixed policy:")
    print_values(V, g)



                
            



