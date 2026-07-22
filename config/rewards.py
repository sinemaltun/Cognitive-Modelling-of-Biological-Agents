REWARDS = {
    "step": -0.1,

    "collect_token": 10,

    "safe_escape": 50,
    "survive_trial": 20,

    # Foraging shaping
    "move_towards_token": 0.5,
    "foraging_predator_distance": 3.0, # Controls strength of the shaping

    # Chase shaping
    "move_towards_predator": -1.0,
    "move_away_from_predator": 0.5,

    "move_towards_safe_zone": 1.0,
    "move_away_from_safe_zone": -1.0,
}

# Tau in the exponential predator-distance potential.
# Controls range of the shaping: Larger values make the predator influence extend farther.
FORAGING_PREDATOR_TAU = 8.0