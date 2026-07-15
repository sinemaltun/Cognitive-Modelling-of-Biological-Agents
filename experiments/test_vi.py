import os
import random
import time
import csv
import pygame
from environment.game_vi import VIGameEnvironment
from agents.value_iteration_agent import ValueIterationAgent
from visualization.pygame_renderer import PygameRenderer

# --- CONFIGURATION ---
TRIALS = 5
THREAT_PROB = 3/5
THREAT_TRIALS = int(TRIALS*THREAT_PROB)
# ---------------------

def wait_and_act(env, agent, renderer, state, duration, message="", is_chase_phase=False):
    """Executes the game visually at 5 frames per second. (1 frame=0.2s)"""
    start_time = time.time()
    print(message) 
    
    status = "SAFE"

    while time.time() < start_time + duration:
        pygame.event.pump()

        try:
            action = agent.choose_action(state, is_chase_phase=is_chase_phase)
        except TypeError:
            action = agent.choose_action(state)
            
        state_next, status, reward = env.step(action, is_chase_phase)
        #State = agent and threat coodinates
        #Status = CAUGHT or SAFE
        #Reward = Total reward so far

        renderer.draw(is_chase=is_chase_phase, status=status) 
        state = state_next
        
        if status == "CAUGHT":
            time.sleep(0.5) #Brief pause to  see the death
            return "CAUGHT", state

        time.sleep(0.2)
    return "SAFE", state
    
       

def run_testing():
    AGENT_TO_TEST = "VI"
    env = VIGameEnvironment()
    agent = ValueIterationAgent()
    renderer = PygameRenderer(env)
    model_path = "models/vi_policy.pkl"
    
    agent.load_model(file_path=model_path)

    time.sleep(2)

    threat_schedule = [True] * THREAT_TRIALS + [False] * (TRIALS - THREAT_TRIALS)
    random.shuffle(threat_schedule)
    results = []

    for trial in range (1, TRIALS+1):
        state = env.reset()

        # --- Phase 1: Foraging ---
        forage_time = random.randint(3, 15)
        phase_1_msg = f"--- Trial {trial}/{TRIALS} ---\nPhase 1: Foraging for {forage_time} seconds..."

        status, state = wait_and_act(env, agent, renderer, state, forage_time, message=phase_1_msg, is_chase_phase=False)

        if status != "CAUGHT":
            threat_appears = threat_schedule[trial-1] #is there a threat in this trial?

            # --- Phase 2: Chasing ---
            if threat_appears:
                #Now the agent has 5 seconds to escap
                chase_time = 5
                phase_2_msg = f"--- Trial {trial}/{TRIALS} ---\n THREAT APPEARED! Chase phase 5 seconds..."
                final_status, state = wait_and_act(env, agent, renderer, state, chase_time, message=phase_2_msg, is_chase_phase=True)

                if final_status != "CAUGHT":
                    if env.player.get_pos() == env.safe_zone.get_pos():
                        final_status = "ESCAPED_IN_BUNKER"
                    else:
                        final_status = "SURVIVED_BY_OUTRUNNING"
            
            else:
                remaining_time = 20-forage_time
                safe_msg = f"--- Trial {trial}/{TRIALS} ---\n Safe trial. Keep collecting tokens..."
                final_status, state = wait_and_act(env, agent, renderer, state, remaining_time, message=safe_msg, is_chase_phase=False)
                if final_status != "CAUGHT":
                    final_status = "SAFE_NO_THREAT_SPAWNED"

        trial_data = {
            "Trial": trial,
            "Tokens_Collected": env.player.tokens,
            "Final_Status": final_status,
            "Total_Steps": env.step_counter
        }
        results.append(trial_data)
        print(f"Trial {trial} finished: {status} | Tokens: {env.player.tokens}")

    renderer.close()
    os.makedirs('results', exist_ok=True)
    file_path = f'results/{AGENT_TO_TEST}_agent_results.csv'
    
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Trial", "Tokens_Collected", "Final_Status", "Total_Steps"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_testing()