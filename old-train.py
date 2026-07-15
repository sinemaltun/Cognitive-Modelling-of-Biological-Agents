# --- TO DOs ---
# 1. Split game runner into high-speed train.py and visual test.py. --done

from utils.libraries import *
from environment.grid import GridEnvironment

from agents.sarsa_agent import SarsaAgent
from agents.qlearning_agent import QLearningAgent 
from agents.random_agent import RandomAgent

# --- CONFIGURATION ---
# change this variable to train different agents ("SARSA", "Q-LEARNING", "RANDOM")
AGENT_TO_TRAIN = "SARSA"
EPOCHS = 80000
# ---------------------

def run_phase_steps(env, agent, state, action, total_steps, is_chase_phase):
    for _ in range(total_steps):
        #1. take a step
        state_next, status, reward = env.step(action, is_chase_phase)
        
        #2. look at the new state and decide the next action
        action_next = agent.choose_action(state_next)
        
        #3. learn from what just happened
        agent.learn(state, action, reward, state_next, action_next)
        
        #4. next state/action becomes the current ones.
        state = state_next
        action = action_next

        if status=="CAUGHT":
            return state, action, "CAUGHT"
        
    return state, action, "SAFE"


def run_training():
    env = GridEnvironment()

    if AGENT_TO_TRAIN == "SARSA":
        agent = SarsaAgent()
        save_path = "models/sarsa_q_table.pkl"

    elif AGENT_TO_TRAIN == "Q-LEARNING":
        agent = QLearningAgent()
        save_path = "models/qlearning_q_table.pkl"

    elif AGENT_TO_TRAIN == "RANDOM":
        agent = RandomAgent()
        save_path = None
    
    else:
        print("Invalid agent selected.")
        return

    print(f"Started training {AGENT_TO_TRAIN} for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS+1):
        state = env.reset()#returns self.get_state()

        #With SARSA, the agent picks its very first action before taking a step
        action = agent.choose_action(state)

        # --- Phase 1: Foraging ---
        #We convert time to steps
        #In our set-up, 0.2 seconds = 1 step
        #Therefore, 1 seconds = 5 steps
        forage_time = random.randint(3, 15)
        forage_steps = int(forage_time/0.2)

        state, action, status = run_phase_steps(env, agent, state, action, forage_steps, is_chase_phase=False)

        # --- Phase 2: Chase or Safe ---
        if status != "CAUGHT":
            #%50 probability while training, so the agent gets to know the threat 
            threat_appears = random.random() < 0.5 

            if threat_appears:
                #Now the agent has 5 seconds to escap
                chase_time = 5
                chase_steps = int(chase_time/0.2)
                state, action, status = run_phase_steps(env, agent, state, action, chase_steps, is_chase_phase=True)
            
            else:
                remaining_time = 20-forage_time
                remaining_steps = int(remaining_time/0.2)
                state, action, status = run_phase_steps(env, agent, state, action, remaining_steps, is_chase_phase=False)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{EPOCHS} complete. (Q-Table size: {len(agent.q_table)} states)")
        
        #Decay epsilon by a tiny fraction every epoch, but never go below 1%
        #The agent starts by exploring 100% of the time and gradually reduces randomness to near-zero 
        # as it gets smarter.
        agent.epsilon = max(0.01, agent.epsilon * 0.9999)
    agent.save_model(file_path=save_path)
          

if __name__ == "__main__":
    run_training()