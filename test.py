# --- TO DOs ---
# 1. Split game runner into high-speed train.py and visual test.py. --done

from utils.libraries import *
from environment.grid import GridEnvironment

from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent 
from agents.random_agent import RandomAgent


# --- CONFIGURATION ---
# change this variable to test different agents ("SARSA", "Q-LEARNING", "RANDOM")
# AGENT_TO_TEST = "SARSA" 
AGENT_TO_TEST = "Q-LEARNING"
TRIALS = 10
THREAT_PROB = 0.2
THREAT_TRIALS = int(TRIALS*THREAT_PROB)
# ---------------------

def wait_and_act(env, agent, duration, message="", is_chase_phase=False):
    """Executes the game visually at 5 frames per second. (1 frame=0.2s)"""
    start_time = time.time()

    #For now, we remove exploration so the agent plays its best
    agent.epsilon = 0.0

    state = env.get_state()

    while time.time() < start_time + duration:
        action = agent.choose_action(state)
        state_next, status, reward = env.step(action, is_chase_phase)
        #State = agent and threat coodinates
        #Status = CAUGHT or SAFE
        #Reward = Total reward so far

        env.render(header_message=message) #trigger the visualizer

        if status == "CAUGHT":
            print("THE AGENT WAS CAUGHT BY THE THREAT")
            return "CAUGHT"

        state = state_next
        
        time.sleep(0.2)
    return "SAFE"
       

def run_testing():
    env = GridEnvironment()

    if AGENT_TO_TEST == "SARSA":
        agent = SarsaAgent()
        model_path = "models/sarsa_q_table.pkl"
    
    elif AGENT_TO_TEST == "Q-LEARNING":
        agent = QLearningAgent()
        model_path = "models/qlearning_q_table.pkl"

    elif AGENT_TO_TEST == "RANDOM":
        agent = RandomAgent()
        model_path = None
    
    else:
        print("Invalid agent selected.")
        return
    
    print(f"Loading {AGENT_TO_TEST} model...")
    agent.load_model(file_path=model_path)

    time.sleep(2)

    threat_schedule = [True] * THREAT_TRIALS + [False] * (TRIALS - THREAT_TRIALS)
    random.shuffle(threat_schedule)

    for trial in range (1, TRIALS+1):
        env.reset()

        # --- Phase 1: Foraging ---
        forage_time = random.randint(3, 15)
        phase_1_msg = f"--- Trial {trial}/{TRIALS} ---\nPhase 1: Foraging for {forage_time} seconds..."

        status = wait_and_act(env, agent, forage_time, message=phase_1_msg, is_chase_phase=False)

        if status != "CAUGHT":
            threat_appears = threat_schedule[trial-1] #is there a threat in this trial?

            if threat_appears:
                #Now the agent has 5 seconds to escap
                chase_time = 5
                phase_2_msg = f"--- Trial {trial}/{TRIALS} ---\n THREAT APPEARED! Chase phase 5 seconds..."
                status = wait_and_act(env,agent, chase_time, message=phase_2_msg, is_chase_phase=True)
            
            else:
                remaining_time = 20-forage_time
                safe_msg = f"--- Trial {trial}/{TRIALS} ---\n Safe trial. Keep collecting tokens..."
                status = wait_and_act(env,agent, remaining_time, message=safe_msg, is_chase_phase=False)

if __name__ == "__main__":
    run_testing()