from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.game import ForagingGame


@dataclass(frozen=True)
class StateFeatures:
    token_dx: int
    token_dy: int

    predator_dx: int
    predator_dy: int

    safe_dx: int
    safe_dy: int

    predator_awake: bool
    phase: int

    @property
    def token_distance(self) -> int:
        return abs(self.token_dx) + abs(self.token_dy)

    @property
    def predator_distance(self) -> int:
        return abs(self.predator_dx) + abs(self.predator_dy)

    @property
    def safe_distance(self) -> int:
        return abs(self.safe_dx) + abs(self.safe_dy)


def build_state_features(env: "ForagingGame",) -> StateFeatures:
    """
    Build a rich feature representation of the current game state.
    Used for reward shaping and feature inspection.
    """

    player = env.player.position
    predator = env.predator.position
    safe = env.safe_zone.position

    nearest_token = min(
        env.tokens,
        key=lambda token: player.manhattan_distance(
            token.position
        ),
    )

    return StateFeatures(
        token_dx=nearest_token.position.x - player.x,
        token_dy=nearest_token.position.y - player.y,

        predator_dx=predator.x - player.x,
        predator_dy=predator.y - player.y,

        safe_dx=safe.x - player.x,
        safe_dy=safe.y - player.y,

        predator_awake=env.predator.awake,
        phase=env.phase.value,
    )


def build_td_state(env: "ForagingGame",) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Build the compact observation used by
    tabular Temporal-Difference methods
    (SARSA and Q-learning).
    """

    features = build_state_features(env)

    return (
        features.token_dx,
        features.token_dy,

        features.predator_dx,
        features.predator_dy,

        features.safe_dx,
        features.safe_dy,

        int(features.predator_awake),
        features.phase,
    )