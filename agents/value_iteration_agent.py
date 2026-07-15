import os
import pickle
import numpy as np
import itertools

class ValueIterationAgent:
    def __init__(self, gamma=0.9, theta=1e-4):
        self.gamma = gamma #discount_factor:how much it cares about future rewards
        self.theta = theta #convergence threshold
        self.policy = {"forage":{}, "chase":{}}

    def train(self, env):
        all_states = env.get_all_states()
        actions = [0,1,2,3,4] #up,right,down,left,stay

        #Training two seperate brains: for foraging and running away
        for is_chase in [False, True]:
            phase_name = "Chase" if is_chase else "Forage"
            print(f"Training VI Policy for {phase_name} Phase...")

            V = {s: 0.0 for s in all_states} #all 16,250 possible game scenarios

            #--- PART 1: Value Iteration (Bellman Eq.)---
            iteration = 0
            while True: #until the values stop changing (converge)
                delta = 0
                for s in all_states:
                    v = V[s]    
                    max_value = float('-inf')

                    for a in actions:
                        transitions = env.get_transition(s,a,is_chase) 
                        expected_value = 0

                        for prob, next_s, reward, is_terminal in transitions:
                            if is_terminal:
                                expected_value += prob * reward
                            else:
                                expected_value += prob * (reward + self.gamma * V[next_s])

                        if expected_value > max_value:
                            max_value = expected_value
                    
                    V[s] = max_value
                    delta = max(delta, abs(v-max_value))
                iteration += 1
                if delta < self.theta: #convergence
                    #print(f"[{phase_name}] Converged in {iteration} iterations.")
                    break

                #--- PART 2: Policy Extraction---
                for s in all_states:
                    best_action = None
                    max_value = float('-inf')

                    for a in actions:
                        transitions = env.get_transition(s,a,is_chase)
                        expected_value = 0

                        for prob, next_s, reward, is_terminal in transitions:
                            if is_terminal:
                                expected_value += prob * reward
                            else:
                                expected_value += prob * (reward + self.gamma * V[next_s])

                        if expected_value > max_value:
                            max_value = expected_value
                            best_action = a

                    self.policy["chase" if is_chase else "forage"][s] = best_action
    
    def choose_action(self,state,is_chase_phase=False):
        """Looks up the pre-calculated perfect move."""
        phase_key = "chase" if is_chase_phase else "forage"
        #If the state exists:return the stored optimal action
        #If the state is missing: return the default value 4 (="Stay" action) as a safe fallback
        return self.policy[phase_key].get(state, 4)
    
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.policy,f)

    def load_model(self,file_path):
        try:
            with open(file_path, 'rb') as f:
                self.policy = pickle.load(f)
        except FileNotFoundError:
            print("No saved model found")