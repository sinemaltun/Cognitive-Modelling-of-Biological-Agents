from dataclasses import dataclass
from typing import TypeAlias

from config.rewards import REWARDS
from environment import Action

from planning.states import (
    ChasePlanningState,
    ForagePlanningState,
)


PlanningState: TypeAlias = (ForagePlanningState | ChasePlanningState)


@dataclass(frozen=True)
class Transition:
    probability: float
    next_state: PlanningState
    reward: float
    terminal: bool


class GridPlanningModel:
    """
    Explicit transition-and-reward model for Value Iteration.

    This does not modify the live ForagingGame. It calculates
    possible outcomes from planning states.
    """

    ACTION_DELTAS = {
        Action.UP: (0, -1),
        Action.RIGHT: (1, 0),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.STAY: (0, 0),
    }

    def __init__(
        self,
        width: int = 24,
        height: int = 16,
        safe_x: int | None = None,
        safe_y: int | None = None,
        predator_speed: int = 2,
        action_noise: float = 0.0,
        caught_penalty: float = -100.0,
        rewards: dict[str, float] | None = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError(
                "Grid dimensions must be positive."
            )

        if predator_speed < 1:
            raise ValueError(
                "predator_speed must be at least 1."
            )

        if not 0.0 <= action_noise <= 1.0:
            raise ValueError(
                "action_noise must be between 0 and 1."
            )

        self.width = width
        self.height = height

        self.safe_x = (
            width - 1
            if safe_x is None
            else safe_x
        )

        self.safe_y = (
            height - 1
            if safe_y is None
            else safe_y
        )

        self.predator_speed = predator_speed
        self.action_noise = action_noise
        self.caught_penalty = caught_penalty

        self.actions = list(Action)

        self.rewards = REWARDS.copy()

        if rewards is not None:
            self.rewards.update(rewards)

    def executed_actions(self, intended_action: Action,) -> list[tuple[float, Action]]:
        """
        Return all actions that may actually be executed.

        With action_noise=0, movement is deterministic.
        Otherwise, noise probability is distributed uniformly
        across the other actions.
        """
        if self.action_noise == 0.0:
            return [(1.0, intended_action)]

        alternative_actions = [
            action
            for action in self.actions
            if action != intended_action
        ]

        alternative_probability = (self.action_noise / len(alternative_actions))

        outcomes = [(1.0 - self.action_noise, intended_action,)]

        outcomes.extend((alternative_probability, action,) for action in alternative_actions)

        return outcomes

    def forage_transitions(self, state: ForagePlanningState, intended_action: Action,) -> list[Transition]:
        transitions = []

        old_token_distance = self._manhattan_distance(
            state.player_x,
            state.player_y,
            state.token_x,
            state.token_y,
        )

        for probability, executed_action in (self.executed_actions(intended_action)):
            next_x, next_y = self._move_player(
                state.player_x,
                state.player_y,
                executed_action,
            )

            new_token_distance = (
                self._manhattan_distance(
                    next_x,
                    next_y,
                    state.token_x,
                    state.token_y,
                )
            )

            reward = self.rewards["step"]

            if new_token_distance < old_token_distance:
                reward += self.rewards[
                    "move_towards_token"
                ]

            elif new_token_distance > old_token_distance:
                reward -= self.rewards[
                    "move_towards_token"
                ]

            collected = (
                next_x == state.token_x
                and next_y == state.token_y
            )

            if collected:
                reward += self.rewards[
                    "collect_token"
                ]

            transitions.append(
                Transition(
                    probability=probability,
                    next_state=ForagePlanningState(
                        player_x=next_x,
                        player_y=next_y,
                        token_x=state.token_x,
                        token_y=state.token_y,
                    ),
                    reward=reward,
                    terminal=collected,
                )
            )

        return transitions

    def chase_transitions(self, state: ChasePlanningState, intended_action: Action,) -> list[Transition]:
        transitions = []

        old_safe_distance = self._manhattan_distance(
            state.player_x,
            state.player_y,
            self.safe_x,
            self.safe_y,
        )

        old_predator_distance = (
            self._manhattan_distance(
                state.player_x,
                state.player_y,
                state.predator_x,
                state.predator_y,
            )
        )

        for probability, executed_action in (
            self.executed_actions(intended_action)
        ):
            player_x, player_y = self._move_player(
                state.player_x,
                state.player_y,
                executed_action,
            )

            new_safe_distance = (
                self._manhattan_distance(
                    player_x,
                    player_y,
                    self.safe_x,
                    self.safe_y,
                )
            )

            new_predator_distance = (
                self._manhattan_distance(
                    player_x,
                    player_y,
                    state.predator_x,
                    state.predator_y,
                )
            )

            reward = self.rewards["step"]

            if new_safe_distance < old_safe_distance:
                reward += self.rewards[
                    "move_towards_safe_zone"
                ]

            elif new_safe_distance > old_safe_distance:
                reward += self.rewards[
                    "move_away_from_safe_zone"
                ]

            if (
                new_predator_distance
                < old_predator_distance
            ):
                reward += self.rewards[
                    "move_towards_predator"
                ]

            elif (
                new_predator_distance
                > old_predator_distance
            ):
                reward += self.rewards.get(
                    "move_away_from_predator",
                    0.0,
                )

            reached_safe_zone = (
                player_x == self.safe_x
                and player_y == self.safe_y
            )

            predator_x = state.predator_x
            predator_y = state.predator_y

            caught = False

            if not reached_safe_zone:
                for _ in range(
                    self.predator_speed
                ):
                    predator_x, predator_y = (
                        self._move_predator_once(
                            predator_x,
                            predator_y,
                            player_x,
                            player_y,
                        )
                    )

                    if (
                        predator_x == player_x
                        and predator_y == player_y
                    ):
                        caught = True
                        break

            if reached_safe_zone:
                reward += self.rewards[
                    "safe_escape"
                ]

            elif caught:
                reward += self.caught_penalty

            transitions.append(
                Transition(
                    probability=probability,
                    next_state=ChasePlanningState(
                        player_x=player_x,
                        player_y=player_y,
                        predator_x=predator_x,
                        predator_y=predator_y,
                    ),
                    reward=reward,
                    terminal=(
                        reached_safe_zone or caught
                    ),
                )
            )

        return transitions

    def generate_forage_states(self,):
        """
        Yield every valid player/target-token combination.

        384 * 383 = 147,072 states on a 24x16 grid.
        """
        for player_x in range(self.width):
            for player_y in range(self.height):
                for token_x in range(self.width):
                    for token_y in range(
                        self.height
                    ):
                        if (
                            player_x == token_x
                            and player_y == token_y
                        ):
                            continue

                        yield ForagePlanningState(
                            player_x=player_x,
                            player_y=player_y,
                            token_x=token_x,
                            token_y=token_y,
                        )

    def generate_chase_states(self,):
        """
        Yield every non-collision player/predator combination.

        384 * 383 = 147,072 states on a 24x16 grid.
        """
        for player_x in range(self.width):
            for player_y in range(self.height):
                for predator_x in range(
                    self.width
                ):
                    for predator_y in range(
                        self.height
                    ):
                        if (
                            player_x == predator_x
                            and player_y == predator_y
                        ):
                            continue

                        yield ChasePlanningState(
                            player_x=player_x,
                            player_y=player_y,
                            predator_x=predator_x,
                            predator_y=predator_y,
                        )

    def _move_player(self, x: int, y: int, action: Action,) -> tuple[int, int]:
        dx, dy = self.ACTION_DELTAS[action]

        next_x = x + dx
        next_y = y + dy

        if (
            0 <= next_x < self.width
            and 0 <= next_y < self.height
        ):
            return next_x, next_y

        return x, y

    @staticmethod
    def _move_predator_once(predator_x: int, predator_y: int, player_x: int, player_y: int,) -> tuple[int, int]:
        if predator_x < player_x:
            predator_x += 1

        elif predator_x > player_x:
            predator_x -= 1

        elif predator_y < player_y:
            predator_y += 1

        elif predator_y > player_y:
            predator_y -= 1

        return predator_x, predator_y

    @staticmethod
    def _manhattan_distance(x1: int, y1: int, x2: int, y2: int,) -> int:
        return abs(x1 - x2) + abs(y1 - y2)