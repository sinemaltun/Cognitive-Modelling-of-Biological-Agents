REWARDS = {
    "step": -0.1,
    "collect_token": 10,
    "caught": -500,
    "safe_escape": 200,
    "survive_trial": 10,
    # Incentivizing certain movements
    "move_towards_token": 1,
    "move_towards_predator": -0.5,
    "move_away_from_predator": 0.2,
    "move_towards_safe_zone": 0.2,
    "move_away_from_safe_zone": -0.1,
}

# TODO: Agent needs more than just information about closest token to really have a choice...
# TODO: More general file for training Q-Learning and SARSA