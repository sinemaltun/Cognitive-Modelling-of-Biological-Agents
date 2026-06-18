#SarsaAgent class

# --- TO DOs ---
# 1.

from utils.libraries import *
env = GridEnvironment()

class SarsaAgent:
    def __init__(self):
        self.q_table = np.zeros((env.height, env.width, 5)) #Q-table
        self.alpha = 0.8 #learning ratel
        self.gamma = 0.9 #discount_factor
        self.epsilon = 0.1 #exploration
        #env.get_state()
        
    def choose_action(self, state):
        action = 0
        if np.random.uniform(0,1) < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[self.state, :])
        
        return action
    
    def learn(self, state, state_next, reward, action, action_next):
        