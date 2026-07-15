from dataclasses import dataclass


@dataclass(frozen=True)
class ForagePlanningState:
    """
    Exact player and current target-token coordinates.

    Collecting the target token terminates this local planning
    problem. The live agent then selects a new target token.
    """

    player_x: int
    player_y: int

    token_x: int
    token_y: int


@dataclass(frozen=True)
class ChasePlanningState:
    """
    Exact player and predator coordinates during chase.
    """

    player_x: int
    player_y: int

    predator_x: int
    predator_y: int