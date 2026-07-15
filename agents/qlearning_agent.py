#QLearningAgent class

# A very helpful source 
# https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9

# --- TO DOs ---
# 1. Build QLearningAgent --done


import os
import pickle
import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99,epsilon=1):
        self.q_table = {} #Q-table
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount_factor
        self.epsilon = epsilon #exploration
        self.actions = [0,1,2,3,4] #up,right,down,left,stay

    def get_q_values(self,state):
        """Fetches Q-values for a state, creating them if the state is new."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(5)
            
        return self.q_table[state]
        
    def choose_action(self, state,is_chase_phase=False):
        """Epsilon-greedy action selection."""
        q_values = self.get_q_values(state)

        #explore
        if np.random.uniform(0,1) < self.epsilon:
            return random.choice(self.actions)
    
        #exploit
        max_val = np.max(q_values)
        best_actions = [act for act in self.actions if q_values[act]==max_val]
        return random.choice(best_actions) #break ties randomly if multiple actions have the same max value


    def learn(self, state, action, reward, state_next, action_next=None):
        """Updates the Q-table using the Q-Learning equation."""

        predict = self.get_q_values(state)[action]

        #Best Q-value for next state
        #Instead of using action_next, Q-learning finds the maximum possible value of the next state
        max_future_q = np.max(self.get_q_values(state_next))
        target = reward + self.gamma * max_future_q

        self.q_table[state][action] = predict + self.alpha * (target - predict)

    def save_model(self, file_path="models/qlearning_q_table.pkl"):
        """Saves the Q-table."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table,f)

    def load_model(self,file_path="models/qlearning_q_table.pkl"):
        """Loads the Q-table."""
        try:
            with open(file_path, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No saved model found")