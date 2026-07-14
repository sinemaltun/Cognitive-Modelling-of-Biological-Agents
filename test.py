from utils.libraries import *
from environment.grid import GridEnvironment
from environment.grid_vi import GridVIEnvironment

from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent 
from agents.random_agent import RandomAgent
from agents.value_iteration import ValueIterationAgent


# --- CONFIGURATION ---
# change this variable to test different agents ("SARSA", "Q-LEARNING", "RANDOM", "VI")
AGENT_TO_TEST = "VI"
TRIALS = 5
THREAT_PROB = 3/5
THREAT_TRIALS = int(TRIALS*THREAT_PROB)
# ---------------------

def wait_and_act(env, agent, duration, message="", is_chase_phase=False):
    """Executes the game visually at 5 frames per second. (1 frame=0.2s)"""
    start_time = time.time()

    #For now, we remove exploration so the agent plays its best
    agent.epsilon = 0.0

    state = env.get_state(is_chase_phase)

    while time.time() < start_time + duration:
        action = agent.choose_action(state,is_chase_phase=is_chase_phase)
        state_next, status, reward = env.step(action, is_chase_phase)
        #State = agent and threat coodinates
        #Status = CAUGHT or SAFE
        #Reward = Total reward so far

        env.render(header_message=message) #trigger the visualizer
        state = state_next
        
        if status == "CAUGHT":
            time.sleep(0.5) #Brief pause to  see the death
            return "CAUGHT"

        time.sleep(0.2)
    return "SAFE"
       

def run_testing():
    if AGENT_TO_TEST == "VI":
        env = GridVIEnvironment()
        agent = ValueIterationAgent()
        model_path = "models/vi_policy.pkl"
    else:
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
    
    #print(f"Loading {AGENT_TO_TEST} model...")
    agent.load_model(file_path=model_path)

    time.sleep(2)

    threat_schedule = [True] * THREAT_TRIALS + [False] * (TRIALS - THREAT_TRIALS)
    random.shuffle(threat_schedule)
    results = []

    for trial in range (1, TRIALS+1):
        env.reset()

        # --- Phase 1: Foraging ---
        forage_time = random.randint(3, 15)
        phase_1_msg = f"--- Trial {trial}/{TRIALS} ---\nPhase 1: Foraging for {forage_time} seconds..."

        status = wait_and_act(env, agent, forage_time, message=phase_1_msg, is_chase_phase=False)

        if status != "CAUGHT":
            threat_appears = threat_schedule[trial-1] #is there a threat in this trial?

            # --- Phase 2: Chasing ---
            if threat_appears:
                #Now the agent has 5 seconds to escap
                chase_time = 5
                phase_2_msg = f"--- Trial {trial}/{TRIALS} ---\n THREAT APPEARED! Chase phase 5 seconds..."
                final_status = wait_and_act(env,agent, chase_time, message=phase_2_msg, is_chase_phase=True)

                if final_status != "CAUGHT":
                    if env.agent_pos == env.safe_zone:
                        final_status = "ESCAPED_IN_BUNKER"
                    else:
                        final_status = "SURVIVED_BY_OUTRUNNING"
            
            else:
                remaining_time = 20-forage_time
                safe_msg = f"--- Trial {trial}/{TRIALS} ---\n Safe trial. Keep collecting tokens..."
                final_status = wait_and_act(env,agent, remaining_time, message=safe_msg, is_chase_phase=False)
                if final_status != "CAUGHT":
                    final_status = "SAFE_NO_THREAT_SPAWNED"

        trial_data = {
            "Trial": trial,
            "Tokens_Collected": env.collected_tokens,
            "Final_Status": final_status,
            "Total_Steps": env.step_counter
        }
        results.append(trial_data)
        print(f"Trial {trial} finished: {status} | Tokens: {env.collected_tokens}")

    with open(f'{AGENT_TO_TEST}_agent_results.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Trial", "Tokens_Collected", "Final_Status", "Total_Steps"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_testing()