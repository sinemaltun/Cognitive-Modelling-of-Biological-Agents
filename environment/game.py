# 1 epoch 10 trials.
# how do we measure time? 
# Let's say 1 second is 10 moves!

# -- Phase 1: Foraging phase --
# Uniform distribution of 3-15 seconds
# If I get 15 seconds there are no threats and I
# just collect tokens in that trial without any consequences
# If less than 15, we move onto the second phase

# -- Phase 2: Chase phase --
# Let's say this epoch is 0.2 probability epoch. 
# Then 2 times out of these 10 trials, a threat will appear.

from libraries import *

# -- EPOCH 1 => THREAT PROB. 0.2 --
threat_prob = 0.2
num_trials = 10
num_events = num_trials*threat_prob
outcomes = [True] * num_events + [False] * (num_trials - num_events)
random.shuffle(outcomes)

print("Starting epoch...")
for trial_number in range(1, num_trials + 1): #Maybe it is a threat-trial or not
    print(f"\n--- Trial {trial_number} ---")

    # Randomly determining how long the foraging phase will be
    forage_phase_time = random.randint(3, 15)
    print(f"Foraging for {forage_phase_time} seconds...")

    # -- Phase 1: Foraging phase --
    end_time = time.time() + forage_phase_time
    while time.time() < end_time
        #Agent will collect tokens here.
        pass

    event_happens = outcomes.pop() #True or False: Will a threat appear?

    if event_happens:
        #Agent will see the threat here
    else:
        #Safe trial... continue until 15 seconds
        
print("\nEpoch finished.")
    

