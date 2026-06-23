#This file will be deleted soon.
# It is replaced with train.py and test.py

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
# 1. Modify print commands for trial number --done
# 2. Integrate grid.spawn_tokens() --done
# 3. Integrate SARSA
# 4. Add a "NEW TRIAL" divider

from utils.libraries import *

def wait_and_act(env, duration, message="", is_chase_phase=False, print_timer=False):
    #1 run of the while loop=1 discrete time step in our environment
    start_time = time.time()

    while time.time() < start_time + duration:
        #PLACEHOLDER
        #For now, the agent chooses a random action (0=Up, 1=Right, 2=Down, 3=Left, 4=Stay)
        #But later, SARSA or Q-learning algorithm will decide this
        action = random.randint(0, 4)

        state, status, reward = env.step(action, is_chase_phase)
        #State = agent and threat coodinates
        #Status = CAUGHT or SAFE
        #Reward = Total reward so far
        
        env.render(header_message=message) #trigger the visualizer

        if status == "CAUGHT":
            print("THE AGENT IS CAUGHT BY THE THREAT")
            return "CAUGHT"

        if print_timer:
            elapsed = int(time.time()-start_time) #How long has it been so far
            print(f"Elapsed: {elapsed}s", end="\r")
        
        # Important note for the concept of time and discrete steps:
        # The functions in this loop run very fast, that is why one run of the loop
        # is almost equal to 0.2
        # 1 run = 0.2 seconds
        # 1 run = 1 action by the agent = 1 turn of the game
        # 1 turn of the game = 0.2 seconds
        time.sleep(0.2)

    if print_timer:
        print(f"Elapsed: {duration}s", end="\r")

    return "SAFE"

# -- EPOCH 1 => THREAT PROB. 0.2 --
threat_prob = 0.2
num_trials = 2
num_events = 1 #int(num_trials*threat_prob)

outcomes = [True] * num_events + [False] * (num_trials - num_events)
random.shuffle(outcomes)

env = GridEnvironment()
print("Starting epoch...")
for trial_number in range(1, num_trials + 1): 
    env.reset()

    #Randomly determining how long the foraging phase will be
    forage_phase_time = random.randint(3, 15)
    
    phase_1_msg = f"--- Trial {trial_number} ---\nPhase 1: Foraging for {forage_phase_time} seconds..."

    # -- Phase 1: Foraging phase --
    status = wait_and_act(env, forage_phase_time,phase_1_msg, is_chase_phase=False, print_timer=False)
    print() 

    event_happens = outcomes.pop() #Will there be a threat?

    if event_happens and status != "CAUGHT":
        # -- Phase 2: Chase phase --
        phase_2_msg = f"--- Trial {trial_number} ---\n THREAT APPEARED! Chase phase 5 seconds..."
        wait_and_act(env, 5, message=phase_2_msg, is_chase_phase=True, print_timer=False)
    else:
        remaining_time = 20-forage_phase_time
        safe_msg = f"--- Trial {trial_number} ---\nSafe trial. Keep collecting tokens..."
        wait_and_act(env, remaining_time, message=safe_msg, is_chase_phase=False, print_timer=False)

print("\nEpoch finished")