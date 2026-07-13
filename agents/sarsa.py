#SarsaAgent class

# A very helpful source 
# https://falabellaindia.com/portfolio/reinforcement-learning-implementation-using-sarsa/

# --- TO DOs ---
# 1. Build SarsaAgent with dictionary Q-table and save/load funcs --done
# 2. Tune gamma so that the agent cares about distanced-tokens (to motivate traveling across the grid)


from utils.libraries import *

class SarsaAgent:
    def __init__(self, alpha=0.1, gamma=0.99,epsilon=1):
        self.q_table = {} #Q-table
        self.alpha = alpha #learning ratel
        self.gamma = gamma #discount_factor:how much it cares about future rewards
        self.epsilon = epsilon #exploration
        self.actions = [0,1,2,3,4] #up,right,down,left,stay

    def get_q_values(self,state):
        """Fetches Q-values for a state, creating them if the state is new."""
        if state not in self.q_table:
            #default unexplored actions to 0.2 so the agent is curious about them
            self.q_table[state] = np.ones(5) * 0.2 
            
        return self.q_table[state]
        
    def choose_action(self, state, is_chase_phase=False):
        """Epsilon-greedy action selection."""
        q_values = self.get_q_values(state)
        # A Q-value is the immediate reward+discounted sum of all the rewards
        # the agent expects to get for the rest of the entire game
        # if it plays perfectly from now on. 
        # --> CHECK "target"

        #explore
        if np.random.uniform(0,1) < self.epsilon:
            return random.choice(self.actions)
    
        #exploit
        max_val = np.max(q_values)
        best_actions = [act for act in self.actions if q_values[act]==max_val]
        return random.choice(best_actions) #break ties randomly if multiple actions have the same max value


    def learn(self, state, action, reward, state_next, action_next):
        """Updates the Q-table using the SARSA equation."""

        # Baseline assumption
        # based on my past experiences, what is the value of taking this specific action in this specific state?
        predict = self.get_q_values(state)[action]

        # Take a step and see
        # target = what the Q-value should've been
        target = reward + self.gamma * self.get_q_values(state_next)[action_next]
        
        # temporal difference error = (target - predict)
        self.q_table[state][action] = predict + self.alpha * (target-predict)

    def save_model(self, file_path="models/sarsa_q_table.pkl"):
        """Saves the Q-table."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table,f)

    def load_model(self,file_path="models/sarsa_q_table.pkl"):
        """Loads the Q-table."""
        try:
            with open(file_path, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No saved model found")