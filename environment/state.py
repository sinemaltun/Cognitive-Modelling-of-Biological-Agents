def build_sarsa_state(env):
    player = env.player.position
    predator = env.predator.position
    safe = env.safe_zone.position

    nearest_token = min(
        env.tokens,
        key=lambda token: player.manhattan_distance(token.position)
    )

    token_dx = nearest_token.position.x - player.x
    token_dy = nearest_token.position.y - player.y

    predator_dx = predator.x - player.x
    predator_dy = predator.y - player.y

    safe_dx = safe.x - player.x
    safe_dy = safe.y - player.y

    return (
        token_dx,
        token_dy,
        predator_dx,
        predator_dy,
        safe_dx,
        safe_dy,
        int(env.predator.awake),
        env.phase.value,
    )