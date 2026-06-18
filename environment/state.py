def build_sarsa_state(env):
    player_pos = env.player.position
    predator_pos = env.predator.position
    safe_pos = env.safe_zone.position

    distance_to_predator = player_pos.manhattan_distance(predator_pos)
    distance_to_safe_zone = player_pos.manhattan_distance(safe_pos)

    if env.tokens:
        distance_to_nearest_token = min(
            player_pos.manhattan_distance(token.position)
            for token in env.tokens
        )
    else:
        distance_to_nearest_token = 0

    return (
        distance_to_predator,
        distance_to_safe_zone,
        distance_to_nearest_token,
        int(env.predator.awake),
        env.phase.value,
    )