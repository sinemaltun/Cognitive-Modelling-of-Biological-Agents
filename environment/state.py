def build_state(env):

    predator = env.predator

    return (
        env.player.x,
        env.player.y,

        predator.x,
        predator.y,

        int(predator.awake),

        env.player.tokens
    )