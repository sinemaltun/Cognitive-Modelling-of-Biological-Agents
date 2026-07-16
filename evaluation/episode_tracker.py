from statistics import mean

from environment import Phase, Status
from environment.state_features import (
    build_state_features,
)

from evaluation.metrics import (
    is_in_safe_quadrant,
)

from evaluation.records import (
    EpisodeRecord,
    StepRecord,
)


class EpisodeTracker:
    def __init__(
        self,
        run_id: str,
        model_type: str,
        mode: str,
        episode: int,
        seed: int | None = None,
    ):
        self.run_id = run_id
        self.model_type = model_type
        self.mode = mode
        self.episode = episode
        self.seed = seed

        self.total_reward = 0.0
        self.total_steps = 0

        self.tokens_collected_gross = 0

        self.predator_distances = []
        self.safe_distances = []

        self.safe_quadrant_steps = 0
        self.distance_travelled = 0

        self.foraging_steps = 0
        self.chase_steps = 0

        self.chase_started = False
        self.chase_start_safe_distance = None
        self.chase_start_predator_distance = None

        self.noisy_action_count = 0

        self.previous_position = None

    def start(self, env) -> None:
        self.previous_position = (env.player.position)

    def record_step(self, env, reward: float, done: bool, info: dict,) -> StepRecord:
        self.total_steps += 1
        self.total_reward += reward

        player = env.player.position
        predator = env.predator.position
        safe = env.safe_zone.position

        features = build_state_features(env)

        nearest_token = min(
            env.tokens,
            key=lambda token:
                player.manhattan_distance(token.position),
        )

        if info["token_collected_this_step"]:
            self.tokens_collected_gross += 1

        if self.previous_position is not None:
            self.distance_travelled += (player.manhattan_distance(self.previous_position))

        self.previous_position = player

        self.predator_distances.append(features.predator_distance)

        self.safe_distances.append(features.safe_distance)

        in_safe_quadrant = (is_in_safe_quadrant(env))

        if in_safe_quadrant:
            self.safe_quadrant_steps += 1

        if info["phase_before"] == Phase.FORAGING:
            self.foraging_steps += 1

        elif info["phase_before"] == Phase.CHASE:
            self.chase_steps += 1

        if info["chase_started_this_step"]:
            self.chase_started = True

            self.chase_start_safe_distance = features.safe_distance

            self.chase_start_predator_distance = features.predator_distance

        if info["action_was_noisy"]:
            self.noisy_action_count += 1

        return StepRecord(
            run_id = self.run_id,
            model_type = self.model_type,
            mode = self.mode,

            episode = self.episode,
            step = self.total_steps,

            phase_before = info["phase_before"].name,

            phase_after = info["phase_after"].name,

            elapsed_time = info["elapsed_time"],
            phase_elapsed_time = (env.phase_elapsed_time()),

            player_x = player.x,
            player_y = player.y,

            predator_x = predator.x,
            predator_y = predator.y,

            safe_x = safe.x,
            safe_y = safe.y,

            nearest_token_x = nearest_token.position.x,

            nearest_token_y = nearest_token.position.y,

            token_distance = features.token_distance,

            predator_distance = features.predator_distance,

            safe_distance = features.safe_distance,

            intended_action = info["intended_action"].name,

            executed_action = info["executed_action"].name,

            action_was_noisy = info["action_was_noisy"],

            moved = info["moved"],

            reward=reward,
            cumulative_reward= self.total_reward,

            tokens_collected_gross=self.tokens_collected_gross,

            tokens_retained = env.player.collected_tokens,

            token_collected_this_step = info["token_collected_this_step"],

            in_safe_quadrant = in_safe_quadrant,

            chase_started_this_step = info["chase_started_this_step"],

            status=info["status"].name,
            done=done,

            threat_probability = env.threat_probability,

            threat_will_appear = env.threat_will_appear,
        )

    def finish(
        self,
        env,
        info: dict,
        epsilon: float | None = None,
        q_table_states: int | None = None,
        policy_states: int | None = None,
    ) -> EpisodeRecord:
        elapsed = info["elapsed_time"]

        safe_quadrant_time = (
            self.safe_quadrant_steps
            * env.dt
        )

        tokens_per_second = (
            self.tokens_collected_gross
            / elapsed
            if elapsed > 0
            else 0.0
        )

        movement_speed = (
            self.distance_travelled / elapsed
            if elapsed > 0
            else 0.0
        )

        safe_fraction = (
            self.safe_quadrant_steps
            / self.total_steps
            if self.total_steps > 0
            else 0.0
        )

        noisy_fraction = (
            self.noisy_action_count
            / self.total_steps
            if self.total_steps > 0
            else 0.0
        )

        status = info["status"]

        return EpisodeRecord(
            run_id = self.run_id,
            model_type = self.model_type,
            mode = self.mode,

            episode = self.episode,
            seed = self.seed,

            threat_probability = env.threat_probability,

            threat_will_appear = env.threat_will_appear,

            forage_duration = env.forage_duration,

            trial_duration = env.trial_duration,

            chase_duration = env.chase_duration,

            total_reward = self.total_reward,
            total_steps = self.total_steps,
            elapsed_time = elapsed,

            tokens_collected_gross = self.tokens_collected_gross,

            tokens_retained = env.player.collected_tokens,

            tokens_per_second = tokens_per_second,

            status = status.name,
            survived = status != Status.CAUGHT,
            caught = status == Status.CAUGHT,
            escaped = status == Status.SAFE,
            timeout = status == Status.TIMEOUT,

            mean_predator_distance=(
                mean(self.predator_distances)
                if self.predator_distances
                else 0.0
            ),

            minimum_predator_distance=(
                min(self.predator_distances)
                if self.predator_distances
                else 0
            ),

            mean_safe_distance=(
                mean(self.safe_distances)
                if self.safe_distances
                else 0.0
            ),

            minimum_safe_distance=(
                min(self.safe_distances)
                if self.safe_distances
                else 0
            ),

            time_in_safe_quadrant = safe_quadrant_time,

            fraction_in_safe_quadrant = safe_fraction,

            distance_travelled = self.distance_travelled,

            movement_speed = movement_speed,

            foraging_steps = self.foraging_steps,

            chase_steps = self.chase_steps,

            chase_started = self.chase_started,

            chase_start_safe_distance = self.chase_start_safe_distance,

            chase_start_predator_distance = self.chase_start_predator_distance,

            noisy_action_count = self.noisy_action_count,

            noisy_action_fraction = noisy_fraction,

            epsilon = epsilon,
            q_table_states = q_table_states,
            policy_states = policy_states,
        )