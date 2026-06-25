import random
import time
from enum import Enum

from entities import (
    Position,
    Player,
    Predator,
    Token,
    SafeZone,
)

from environment.grid import Grid
from environment.state import build_sarsa_state
from config.rewards import REWARDS


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
    ):
        self.grid = Grid(width, height)

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

        self.rewards = REWARDS.copy()
        if rewards is not None:
            self.rewards.update(rewards)

        self.phase = Phase.FORAGING
        self.status = Status.RUNNING

        self.reset()

    def reset(self):
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
            random.random() < self.threat_probability
        )

        self.safe_zone = SafeZone(
            Position(self.grid.width - 1, self.grid.height - 1)
        )

        self.predator = Predator(
            position=Position(0, 0),
            awake=False,
            speed=self.predator_speed,
        )

        self.player = Player(
            position=self._random_empty_position(
                forbidden={
                    self.safe_zone.position,
                    self.predator.position,
                }
            )
        )

        self.tokens = []
        self.spawn_tokens(self.initial_tokens)

        return self.get_state()

    def get_state(self):
        return build_sarsa_state(self)

    def elapsed_time(self) -> float:
        if self.realtime:
            return time.time() - self.start_time

        return self.simulated_time

    def phase_elapsed_time(self) -> float:
        if self.realtime:
            return time.time() - self.phase_start_time

        return self.phase_simulated_time

    def remaining_trial_time(self) -> float:
        return max(
            0.0,
            self.trial_duration - self.elapsed_time()
        )

    def remaining_chase_time(self) -> float:
        if self.phase != Phase.CHASE:
            return 0.0

        return max(
            0.0,
            self.chase_duration - self.phase_elapsed_time()
        )

    def step(self, action):
        if self.phase == Phase.FINISHED:
            return self.get_state(), 0, True, self._info()

        if isinstance(action, int):
            action = Action(action)

        if not self.realtime:
            self.simulated_time += self.dt
            self.phase_simulated_time += self.dt

        reward = self.rewards["step"]

        old_distance = self._distance_to_nearest_token()

        reward += self._move_player(action)

        new_distance = self._distance_to_nearest_token()

        if new_distance < old_distance:
            reward += self.rewards["move_towards_token"]
        elif new_distance > old_distance:
            reward -= self.rewards["move_towards_token"]

        reward += self._collect_token_if_present()

        self._maybe_start_chase()

        if self.phase == Phase.CHASE:
            reward += self._move_predator()

        reward += self._check_terminal_conditions()

        done = self.phase == Phase.FINISHED

        return self.get_state(), reward, done, self._info()

    def _move_player(self, action: Action) -> int:
        dx, dy = ACTION_DELTAS[action]

        new_position = self.player.position.moved(dx, dy)

        if self.grid.inside(new_position):
            self.player.move_to(new_position)
            return 0

        return self.rewards["hit_wall"]

    def _collect_token_if_present(self) -> int:
        for token in list(self.tokens):
            if token.position == self.player.position:
                self.tokens.remove(token)
                self.player.collect_token()
                self.spawn_tokens(1)
                return self.rewards["collect_token"]

        return 0

    def _distance_to_nearest_token (self) -> int:

        player = self.player.position

        return min(
            player.manhattan_distance(token.position)
            for token in self.tokens
        )

    def _maybe_start_chase(self) -> None:
        if self.phase != Phase.FORAGING:
            return

        forage_time_over = (
            self.phase_elapsed_time() >= self.forage_duration
        )

        if forage_time_over and self.threat_will_appear:
            self.phase = Phase.CHASE
            self.status = Status.RUNNING
            self.phase_start_time = time.time()
            self.phase_simulated_time = 0.0
            self.predator.wake_up()

    def _move_predator(self) -> int:
        for _ in range(self.predator.speed):
            self.predator.move_towards(self.player.position)

            if self.predator.catches(
                self.player.position,
                self.safe_zone.position,
            ):
                self.player.lose_tokens()
                self.status = Status.CAUGHT
                self.phase = Phase.FINISHED
                return self.rewards["caught"]

        return 0

    def _check_terminal_conditions(self) -> int:
        if self.status == Status.CAUGHT:
            return 0

        if self.safe_zone.contains(self.player.position):
            if self.phase == Phase.CHASE:
                self.status = Status.SAFE
                self.phase = Phase.FINISHED
                return self.rewards["safe_escape"]

        if self.phase == Phase.CHASE:
            if self.phase_elapsed_time() >= self.chase_duration:
                self.status = Status.SAFE
                self.phase = Phase.FINISHED
                return self.rewards["safe_escape"]

        if self.elapsed_time() >= self.trial_duration:
            self.status = Status.TIMEOUT
            self.phase = Phase.FINISHED
            return self.rewards["survive_trial"]

        return 0

    def spawn_tokens(self, amount: int = 1) -> None:
        forbidden = {
            self.safe_zone.position,
            self.predator.position,
            self.player.position,
        }

        forbidden.update(
            token.position for token in self.tokens
        )

        for _ in range(amount):
            position = self._random_empty_position(forbidden)
            self.tokens.append(Token(position))
            forbidden.add(position)

    def _random_empty_position(self, forbidden: set[Position]) -> Position:
        while True:
            position = Position(
                random.randint(0, self.grid.width - 1),
                random.randint(0, self.grid.height - 1),
            )

            if position not in forbidden:
                return position

    def _info(self) -> dict:
        return {
            "phase": self.phase,
            "status": self.status,
            "tokens_collected": self.player.collected_tokens,
            "elapsed_time": self.elapsed_time(),
            "remaining_trial_time": self.remaining_trial_time(),
            "remaining_chase_time": self.remaining_chase_time(),
            "threat_will_appear": self.threat_will_appear,
            "forage_duration": self.forage_duration,
            "realtime": self.realtime,
        }