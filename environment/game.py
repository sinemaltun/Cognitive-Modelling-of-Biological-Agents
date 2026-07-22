import math
import random
import time
from enum import Enum

from config.rewards import (
    FORAGING_PREDATOR_TAU,
    REWARDS,
)

from entities import (
    Player,
    Position,
    Predator,
    SafeZone,
    Token,
)

from environment.grid import Grid

from environment.state_features import (
    build_state_features,
    build_td_state,
)


class Phase(Enum):
    FORAGING = 0
    CHASE = 1
    FINISHED = 2


class Status(Enum):
    RUNNING = 0
    SAFE = 1
    CAUGHT = 2
    TIMEOUT = 3


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


ACTION_DELTAS = {
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.STAY: (0, 0),
}


class ForagingGame:
    def __init__(
        self,
        width: int = 24,
        height: int = 16,
        trial_duration: float = 20.0,
        chase_duration: float = 5.0,
        min_forage_time: float = 3.0,
        max_forage_time: float = 15.0,
        initial_tokens: int = 11,
        threat_probability: float = 0.2,
        predator_speed: int = 2,
        rewards: dict | None = None,
        realtime: bool = True,
        steps_per_second: int = 10,
        action_noise: float = 0.0,
    ):
        if width <= 0:
            raise ValueError("width must be greater than zero.")

        if height <= 0:
            raise ValueError("height must be greater than zero.")

        if trial_duration <= 0:
            raise ValueError("trial_duration must be greater than zero.")

        if chase_duration <= 0:
            raise ValueError("chase_duration must be greater than zero.")

        if min_forage_time < 0:
            raise ValueError("min_forage_time cannot be negative.")

        if max_forage_time < min_forage_time:
            raise ValueError("max_forage_time must be greater than or equal to min_forage_time.")

        if initial_tokens < 1:
            raise ValueError("initial_tokens must be at least one.")

        if not 0.0 <= threat_probability <= 1.0:
            raise ValueError("threat_probability must be between 0 and 1.")

        if predator_speed < 1:
            raise ValueError("predator_speed must be at least one.")

        if steps_per_second <= 0:
            raise ValueError("steps_per_second must be greater than zero.")

        if not 0.0 <= action_noise <= 1.0:
            raise ValueError("action_noise must be between 0 and 1.")

        self.grid = Grid(width=width,height=height,)

        self.trial_duration = trial_duration
        self.chase_duration = chase_duration

        self.min_forage_time = min_forage_time
        self.max_forage_time = max_forage_time

        self.initial_tokens = initial_tokens
        self.threat_probability = threat_probability
        self.predator_speed = predator_speed

        self.realtime = realtime
        self.steps_per_second = steps_per_second
        self.dt = 1.0 / steps_per_second

        self.action_noise = action_noise

        self.rewards = REWARDS.copy()

        if rewards is not None:
            self.rewards.update(rewards)

        self.phase = Phase.FORAGING
        self.status = Status.RUNNING

        self.reset()

    def reset(self):
        """
        Start a new trial and return its initial TD state.
        """
        self.phase = Phase.FORAGING
        self.status = Status.RUNNING

        self.start_time = time.time()
        self.phase_start_time = self.start_time

        self.simulated_time = 0.0
        self.phase_simulated_time = 0.0

        self.forage_duration = random.uniform(
            self.min_forage_time,
            self.max_forage_time,
        )

        self.threat_will_appear = (
            random.random()
            < self.threat_probability
        )

        self.safe_zone = SafeZone(
            Position(
                self.grid.width - 1,
                self.grid.height - 1,
            )
        )

        self.predator = Predator(
            position=Position(0, 0),
            awake=False,
            speed=self.predator_speed,
        )

        self.player = Player(
            position=self._random_empty_position(forbidden={self.safe_zone.position, self.predator.position,}
            )
        )

        self.tokens: list[Token] = []
        self.spawn_tokens(self.initial_tokens)

        # Information about the most recently executed step.
        #
        # These fields are initialized here so that _info()
        # remains safe to call immediately after reset().
        self.last_intended_action: Action | None = None
        self.last_executed_action: Action | None = None

        self.last_action_was_noisy = False
        self.last_move_succeeded = False
        self.last_token_collected = False
        self.last_chase_started = False

        self.last_chase_start_safe_distance: int | None = None
        self.last_chase_start_predator_distance: int | None = None

        self.last_phase_before = self.phase

        return self.get_state()

    def get_state(self):
        """
        Return the observation used by the TD agents.
        """
        return build_td_state(self)

    def elapsed_time(self) -> float:
        if self.realtime:
            return time.time() - self.start_time

        return self.simulated_time

    def phase_elapsed_time(self) -> float:
        if self.realtime:
            return time.time() - self.phase_start_time

        return self.phase_simulated_time

    def remaining_trial_time(self) -> float:
        return max(0.0, self.trial_duration - self.elapsed_time(),)

    def remaining_chase_time(self) -> float:
        if self.phase != Phase.CHASE:
            return 0.0

        return max(0.0,self.chase_duration - self.phase_elapsed_time(),)

    def step(self, action: Action | int,):
        """
        Advance the environment by one action.

        Returns:
            next_state:
                TD observation after the transition.

            reward:
                Reward generated by this transition.

            done:
                True when the trial has finished.

            info:
                Additional information for rendering, evaluation,
                and behavioral logging.
        """
        if self.phase == Phase.FINISHED:
            return (self.get_state(), 0.0, True, self._info(),)

        if isinstance(action, int):
            action = Action(action)

        if not isinstance(action, Action):
            raise TypeError("action must be an Action or integer action value.")

        phase_before = self.phase

        intended_action = action

        executed_action = self._execute_action(intended_action)

        old_position = self.player.position

        # Reset one-step event flags before processing this action.
        self.last_token_collected = False
        self.last_chase_started = False

        self.last_chase_start_safe_distance = None
        self.last_chase_start_predator_distance = None

        if not self.realtime:
            self.simulated_time += self.dt
            self.phase_simulated_time += self.dt

        reward = self.rewards["step"]

        old_features = build_state_features(self)

        self._move_player(executed_action)

        moved = (self.player.position != old_position)

        new_features = build_state_features(self)

        # -----------------------------------------------------
        # Foraging reward shaping
        # -----------------------------------------------------

        if phase_before == Phase.FORAGING:
            if new_features.token_distance < old_features.token_distance:
                reward += self.rewards["move_towards_token"]

            elif new_features.token_distance > old_features.token_distance:
                reward -= self.rewards["move_towards_token"]

            reward += self._foraging_predator_shaping(
                old_distance=(
                    old_features.predator_distance
                ),
                new_distance=(
                    new_features.predator_distance
                ),
            )

        # -----------------------------------------------------
        # Chase reward shaping
        # -----------------------------------------------------

        elif phase_before == Phase.CHASE:
            if new_features.predator_distance < old_features.predator_distance:
                reward += self.rewards["move_towards_predator"]

            elif new_features.predator_distance > old_features.predator_distance:
                reward += self.rewards.get("move_away_from_predator",0.0,)

            if new_features.safe_distance < old_features.safe_distance:
                reward += self.rewards["move_towards_safe_zone"]

            elif new_features.safe_distance > old_features.safe_distance:
                reward += self.rewards["move_away_from_safe_zone"]

        reward += self._collect_token_if_present()

        self._maybe_start_chase()

        chase_started = phase_before == Phase.FORAGING and self.phase == Phase.CHASE

        if chase_started:
            chase_features = build_state_features(self)

            self.last_chase_start_safe_distance = chase_features.safe_distance

            self.last_chase_start_predator_distance = chase_features.predator_distance

        if self.phase == Phase.CHASE:
            reward += self._move_predator()

        reward += self._check_terminal_conditions()

        done = self.phase == Phase.FINISHED

        # Store transition information for the logger.
        self.last_intended_action = intended_action
        self.last_executed_action = executed_action

        self.last_action_was_noisy = intended_action != executed_action

        self.last_move_succeeded = moved
        self.last_chase_started = chase_started
        self.last_phase_before = phase_before

        return (self.get_state(), reward, done, self._info(),)

    def _execute_action(self, intended_action: Action,) -> Action:
        """
        Apply motor noise to an intended action.

        action_noise is the exact probability that an action other
        than the intended action is executed.
        """
        if self.action_noise == 0.0:
            return intended_action

        if random.random() >= self.action_noise:
            return intended_action

        alternatives = [
            action
            for action in Action
            if action != intended_action
        ]

        return random.choice(alternatives)

    def _move_player(self, action: Action,) -> None:
        """
        Apply an executed action to the player.

        If the action would leave the grid, the player remains in
        the current cell.
        """
        dx, dy = ACTION_DELTAS[action]

        new_position = self.player.position.moved(dx,dy,)

        if self.grid.inside(new_position):
            self.player.move_to(new_position)

    def _collect_token_if_present(self) -> float:
        """
        Collect a token from the player's current position.

        A replacement token is spawned immediately so the number
        of tokens on the grid stays constant.
        """
        for token in list(self.tokens):
            if token.position != self.player.position:
                continue

            self.tokens.remove(token)
            self.player.collect_token()

            self.last_token_collected = True

            self.spawn_tokens(1)

            return self.rewards["collect_token"]

        return 0.0

    def _foraging_predator_shaping(self, old_distance: int,new_distance: int,) -> float:
        """
        Reward movement away from the sleeping predator.

        The potential saturates exponentially, so movement matters
        most when the player is close to the predator and has
        little effect once the player is already far away.
        """
        old_potential = (1.0 - math.exp( -old_distance / FORAGING_PREDATOR_TAU))

        new_potential = (1.0 - math.exp(-new_distance / FORAGING_PREDATOR_TAU))

        return self.rewards["foraging_predator_distance"] * (new_potential - old_potential)

    def _maybe_start_chase(self) -> None:
        """
        Switch from foraging to chase when the randomized forage
        period has ended and this is a threat trial.
        """
        if self.phase != Phase.FORAGING:
            return

        forage_time_over = self.phase_elapsed_time()>= self.forage_duration

        if forage_time_over and self.threat_will_appear:
            self.phase = Phase.CHASE
            self.status = Status.RUNNING

            self.phase_start_time = time.time()
            self.phase_simulated_time = 0.0

            self.predator.wake_up()

    def _move_predator(self) -> float:
        """
        Move the predator toward the player and detect capture.

        When caught, the reward value of all currently retained
        tokens is cancelled.
        """
        for _ in range(self.predator.speed):
            self.predator.move_towards(self.player.position)

            caught = self.predator.catches(self.player.position, self.safe_zone.position,)

            if not caught:
                continue

            lost_token_reward = (self.player.collected_tokens * self.rewards["collect_token"])

            self.player.lose_tokens()

            self.status = Status.CAUGHT
            self.phase = Phase.FINISHED

            return -float(lost_token_reward)

        return 0.0

    def _check_terminal_conditions(self) -> float:
        """
        Check whether the current trial should finish.
        """
        if self.status == Status.CAUGHT:
            return 0.0

        if self.phase == Phase.CHASE and self.safe_zone.contains(self.player.position):
            self.status = Status.SAFE
            self.phase = Phase.FINISHED

            return self.rewards["safe_escape"]

        if self.phase == Phase.CHASE and self.phase_elapsed_time() >= self.chase_duration:
            self.status = Status.SAFE
            self.phase = Phase.FINISHED

            return self.rewards["safe_escape"]

        if self.elapsed_time() >= self.trial_duration:
            self.status = Status.TIMEOUT
            self.phase = Phase.FINISHED

            return self.rewards["survive_trial"]

        return 0.0

    def spawn_tokens(self, amount: int = 1,) -> None:
        """
        Spawn tokens in distinct currently unoccupied cells.
        """
        forbidden = {
            self.safe_zone.position,
            self.predator.position,
            self.player.position,
        }

        forbidden.update(
            token.position
            for token in self.tokens
        )

        available_cells = self.grid.width * self.grid.height - len(forbidden)

        if amount > available_cells:
            raise ValueError("Not enough empty cells to spawn " f"{amount} token(s).")

        for _ in range(amount):
            position = self._random_empty_position(forbidden)

            self.tokens.append(Token(position))

            forbidden.add(position)

    def _random_empty_position(self,forbidden: set[Position],) -> Position:
        """
        Return a random position that is not forbidden.
        """
        while True:
            position = Position(
                random.randint(
                    0,
                    self.grid.width - 1,
                ),
                random.randint(
                    0,
                    self.grid.height - 1,
                ),
            )

            if position not in forbidden:
                return position

    def _info(self) -> dict:
        """
        Return auxiliary information about the current trial and
        the most recently executed transition.
        """
        return {
            "phase": self.phase,
            "phase_before": self.last_phase_before,
            "phase_after": self.phase,

            "status": self.status,

            "tokens_collected": self.player.collected_tokens,

            "elapsed_time": self.elapsed_time(),

            "phase_elapsed_time": self.phase_elapsed_time(),

            "remaining_trial_time": self.remaining_trial_time(),

            "remaining_chase_time": self.remaining_chase_time(),

            "threat_probability": self.threat_probability,

            "threat_will_appear": self.threat_will_appear,

            "forage_duration": self.forage_duration,

            "realtime": self.realtime,

            "intended_action": self.last_intended_action,

            "executed_action": self.last_executed_action,

            "action_was_noisy": self.last_action_was_noisy,

            "moved": self.last_move_succeeded,

            "token_collected_this_step": self.last_token_collected,

            "chase_started_this_step": self.last_chase_started,
            
            "chase_start_safe_distance": self.last_chase_start_safe_distance,

            "chase_start_predator_distance": self.last_chase_start_predator_distance,
        }