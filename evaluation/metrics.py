from environment import Action


OPPOSITE_ACTIONS = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
    Action.STAY: Action.STAY,
}


def is_in_safe_quadrant(env) -> bool:
    """
    Return whether the player is in the quarter of the grid
    containing the safe zone.
    """

    player = env.player.position
    safe = env.safe_zone.position

    horizontal_middle = env.grid.width // 2
    vertical_middle = env.grid.height // 2

    if safe.x >= horizontal_middle:
        correct_horizontal_half = (player.x >= horizontal_middle)
    else:
        correct_horizontal_half = (player.x < horizontal_middle)

    if safe.y >= vertical_middle:
        correct_vertical_half = (player.y >= vertical_middle)
    else:
        correct_vertical_half = (player.y < vertical_middle)

    return correct_horizontal_half and correct_vertical_half


def is_action_reversal(previous_action: Action | None, current_action: Action,) -> bool:
    if previous_action is None:
        return False

    return OPPOSITE_ACTIONS[previous_action] == current_action