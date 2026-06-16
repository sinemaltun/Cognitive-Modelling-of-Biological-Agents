# --- SET-UP EXPLANATION ---
# 1 epoch 10 trials.
# how do we measure time? --Implementing seconds
# I define a trial to be 20 seconds

# -- Phase 1: Foraging phase --
# Uniform distribution of 3-15 seconds
# If I get 15 seconds there are no threats and I
# just collect tokens in that trial without any consequences
# If less than 15, we move onto the second phase

# -- Phase 2: Chase phase --
# Let's say this epoch is a 0.2 probability epoch. 
# Then 2 times out of these 10 trials, a threat will appear.

# --- TO DOs ---
# 1. Integrate grid.spawn_tokens() --done (inside env.step)
# 2. Integrate SARSA

from libraries import *

def wait_and_act(env, duration, is_chase_phase=False, print_timer=False):
    start_time = time.time()

    while time.time() < start_time + duration:
        #PLACEHOLDER
        #For now, the agent chooses a random action (0=Up, 1=Right, 2=Down, 3=Left)
        #But later, SARSA or Q-learning algorithm will decide this
        action = random.randint(0, 3)

        state, status, reward = env.step(action, is_chase_phase)
        #State = agent and threat coodinates
        #Status = CAUGHT or SAFE
        #Reward = Total reward so far
        
        env.render() #trigger the visualizer

        if status == "CAUGHT":
            print("THE AGENT IS CAUGHT BY THE THREAT")
            return "CAUGHT"

        if print_timer:
            elapsed = int(time.time()-start_time) #How long has it been so far
            print(f"Elapsed: {elapsed}s", end="\r")
        
        time.sleep(0.2)

    if print_timer:
        print(f"Elapsed: {duration}s", end="\r")

    return "SAFE"

# -- EPOCH 1 => THREAT PROB. 0.2 --
threat_prob = 0.2
num_trials = 5
num_events = int(num_trials*threat_prob)

outcomes = [True] * num_events + [False] * (num_trials - num_events)
random.shuffle(outcomes)

env = GridEnvironment()
print("Starting epoch...")
for trial_number in range(1, num_trials + 1): 
    print(f"\n--- Trial {trial_number} ---")

    env.reset()

    #Randomly determining how long the foraging phase will be
    forage_phase_time = random.randint(3, 15)
    print(f"Foraging for {forage_phase_time} seconds...")

    # -- Phase 1: Foraging phase --
    status = wait_and_act(env, forage_phase_time, is_chase_phase=False, print_timer=False)
    print() 

    event_happens = outcomes.pop() #Will there be a threat?

    if event_happens and status != "CAUGHT":
        # -- Phase 2: Chase phase --
        print("Threat appeared. Chase phase 5 seconds...")
        wait_and_act(env, 5, is_chase_phase=True, print_timer=False)
    else:
        remainig_time = 20-forage_phase_time
        print("Safe trial. Keep collecting tokens...")
        wait_and_act(env, remainig_time,is_chase_phase=False, print_timer=False)    

print("\nEpoch finished")