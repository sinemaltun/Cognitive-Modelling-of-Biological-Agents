#RandomAgent class

# --- TO DOs ---
# 1. Build RandomAgent --done

from utils.libraries import *

class RandomAgent:
    def __init__(self):
        self.q_table = {}
        self.actions = [0,1,2,3,4] #up,right,down,left,stay
        
    def choose_action(self, state):
        """Picks an action completely at random."""
        action = random.choice(self.actions)

    def learn(self, state, action, reward, state_next, action_next=None):
        """A random agent does not learn."""
        pass

    def save_model(self, file_path=None):
        """No memory to save."""
        pass

    def load_model(self,file_path="models/qlearning_q_table.pkl"):
        """No memory to load."""
        pass