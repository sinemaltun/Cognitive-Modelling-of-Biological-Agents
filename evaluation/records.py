from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class StepRecord:
    run_id: str
    model_type: str
    mode: str

    episode: int
    step: int

    phase_before: str
    phase_after: str

    elapsed_time: float
    phase_elapsed_time: float

    player_x: int
    player_y: int

    predator_x: int
    predator_y: int

    safe_x: int
    safe_y: int

    nearest_token_x: int
    nearest_token_y: int

    token_distance: int
    predator_distance: int
    safe_distance: int

    intended_action: str
    executed_action: str
    action_was_noisy: bool
    moved: bool

    reward: float
    cumulative_reward: float

    tokens_collected_gross: int
    tokens_retained: int
    token_collected_this_step: bool

    in_safe_quadrant: bool
    chase_started_this_step: bool

    status: str
    done: bool

    threat_probability: float
    threat_will_appear: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRecord:
    run_id: str
    model_type: str
    mode: str

    episode: int

    run_seed: int | None
    episode_seed: int | None

    threat_probability: float
    threat_will_appear: bool
    threat_trial: bool
    survived_threat: bool | None

    forage_duration: float
    trial_duration: float
    chase_duration: float

    total_reward: float
    total_steps: int
    elapsed_time: float

    tokens_collected_gross: int
    tokens_retained: int
    tokens_per_second: float

    status: str

    survived: bool
    caught: bool
    escaped: bool
    timeout: bool

    mean_predator_distance: float
    minimum_predator_distance: int

    mean_safe_distance: float
    minimum_safe_distance: int

    time_in_safe_quadrant: float
    fraction_in_safe_quadrant: float

    distance_travelled: int
    movement_speed: float

    foraging_steps: int
    chase_steps: int
    chase_started: bool

    chase_start_safe_distance: int | None
    chase_start_predator_distance: int | None

    noisy_action_count: int
    noisy_action_fraction: float

    epsilon_used: float | None
    q_table_states: int | None
    policy_states: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingProgressRecord:
    run_id: str
    model_type: str

    episode: int
    window_size: int

    mean_reward: float
    mean_tokens_gross: float
    mean_tokens_retained: float

    survival_rate: float
    threat_trial_count: int
    threat_survival_rate: float | None

    caught_rate: float
    escape_rate: float

    epsilon_used: float
    epsilon_next: float

    q_table_states: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValueIterationProgressRecord:
    run_id: str
    phase: str

    iteration: int
    maximum_value_change: float
    state_count: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)