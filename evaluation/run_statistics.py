from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evaluation.records import EpisodeRecord


@dataclass
class RunStatistics:
    """
    Incrementally aggregate statistics from completed episodes.

    The class stores running totals rather than retaining every
    EpisodeRecord in memory. It can therefore be used for long
    training runs, evaluations, and human sessions.
    """

    episode_count: int = 0

    total_reward: float = 0.0
    total_tokens_gross: int = 0
    total_tokens_retained: int = 0
    total_steps: int = 0
    total_elapsed_time: float = 0.0

    survived_count: int = 0
    caught_count: int = 0
    escaped_count: int = 0
    timeout_count: int = 0

    threat_trial_count: int = 0
    threat_survival_count: int = 0
    threat_caught_count: int = 0

    chase_started_count: int = 0

    chase_start_safe_distance_total: float = 0.0
    chase_start_safe_distance_count: int = 0

    chase_start_predator_distance_total: float = 0.0
    chase_start_predator_distance_count: int = 0

    noisy_action_count: int = 0
    total_recorded_actions: int = 0

    def update(self, episode_record: EpisodeRecord,) -> None:
        """
        Add one completed episode to the running statistics.
        """
        self.episode_count += 1

        self.total_reward += float(episode_record.total_reward)

        self.total_tokens_gross += int(episode_record.tokens_collected_gross)

        self.total_tokens_retained += int(episode_record.tokens_retained)

        self.total_steps += int(episode_record.total_steps)

        self.total_elapsed_time += float(episode_record.elapsed_time)

        self.survived_count += int(episode_record.survived)

        self.caught_count += int(episode_record.caught)

        self.escaped_count += int(episode_record.escaped)

        self.timeout_count += int(episode_record.timeout)

        self.noisy_action_count += int(episode_record.noisy_action_count)

        self.total_recorded_actions += int(episode_record.total_steps)

        if episode_record.threat_trial:
            self.threat_trial_count += 1

            if episode_record.survived_threat is True:
                self.threat_survival_count += 1

            if episode_record.caught:
                self.threat_caught_count += 1

        if episode_record.chase_started:
            self.chase_started_count += 1

        chase_start_safe_distance = episode_record.chase_start_safe_distance

        if chase_start_safe_distance is not None:
            self.chase_start_safe_distance_total += float(chase_start_safe_distance)

            self.chase_start_safe_distance_count += 1

        chase_start_predator_distance = episode_record.chase_start_predator_distance

        if chase_start_predator_distance is not None:
            self.chase_start_predator_distance_total += float(chase_start_predator_distance)

            self.chase_start_predator_distance_count += 1

    @staticmethod
    def _safe_divide(numerator: float | int, denominator: float | int,) -> float | None:
        """
        Return numerator / denominator, or None for a zero denominator.
        """
        if denominator == 0:
            return None

        return float(numerator) / float(denominator)

    @property
    def overall_survival_rate(self) -> float | None:
        return self._safe_divide(self.survived_count, self.episode_count,)

    @property
    def caught_rate(self) -> float | None:
        return self._safe_divide(self.caught_count, self.episode_count,)

    @property
    def escape_rate(self) -> float | None:
        return self._safe_divide(self.escaped_count, self.episode_count,)

    @property
    def timeout_rate(self) -> float | None:
        return self._safe_divide(self.timeout_count, self.episode_count,)

    @property
    def threat_trial_rate(self) -> float | None:
        return self._safe_divide(self.threat_trial_count, self.episode_count,)

    @property
    def threat_survival_rate(self) -> float | None:
        """
        Return survival among threat trials only.

        Non-threat episodes are excluded from the denominator.
        """
        return self._safe_divide(self.threat_survival_count, self.threat_trial_count,)

    @property
    def threat_caught_rate(self) -> float | None:
        return self._safe_divide(self.threat_caught_count, self.threat_trial_count,)

    @property
    def mean_reward(self) -> float | None:
        return self._safe_divide(self.total_reward, self.episode_count,)

    @property
    def mean_tokens_gross(self) -> float | None:
        return self._safe_divide(self.total_tokens_gross, self.episode_count,)

    @property
    def mean_tokens_retained(self) -> float | None:
        return self._safe_divide(self.total_tokens_retained, self.episode_count,)

    @property
    def mean_steps(self) -> float | None:
        return self._safe_divide(self.total_steps, self.episode_count,)

    @property
    def mean_elapsed_time(self) -> float | None:
        return self._safe_divide(self.total_elapsed_time, self.episode_count,)

    @property
    def noisy_action_fraction(self) -> float | None:
        return self._safe_divide(self.noisy_action_count, self.total_recorded_actions,)

    @property
    def mean_chase_start_safe_distance(self,) -> float | None:
        return self._safe_divide(self.chase_start_safe_distance_total, self.chase_start_safe_distance_count,)

    @property
    def mean_chase_start_predator_distance(self,) -> float | None:
        return self._safe_divide(self.chase_start_predator_distance_total, self.chase_start_predator_distance_count,)

    def to_summary_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable summary of the complete run.
        """
        return {
            "episodes_completed": self.episode_count,

            "outcomes": {
                "survived_count": self.survived_count,
                "caught_count": self.caught_count,
                "escape_count": self.escaped_count,
                "timeout_count": self.timeout_count,

                "overall_survival_rate": self.overall_survival_rate,

                "caught_rate": self.caught_rate,
                "escape_rate": self.escape_rate,
                "timeout_rate": self.timeout_rate,
            },

            "threat_trials": {
                "threat_trial_count": self.threat_trial_count,

                "non_threat_trial_count": self.episode_count - self.threat_trial_count,

                "threat_trial_rate": self.threat_trial_rate,

                "threat_survival_count": self.threat_survival_count,

                "threat_caught_count": self.threat_caught_count,

                "threat_survival_rate": self.threat_survival_rate,

                "threat_caught_rate": self.threat_caught_rate,
            },

            "performance": {
                "total_reward": self.total_reward,
                "mean_reward": self.mean_reward,

                "total_tokens_gross": self.total_tokens_gross,

                "mean_tokens_gross": self.mean_tokens_gross,

                "total_tokens_retained": self.total_tokens_retained,

                "mean_tokens_retained": self.mean_tokens_retained,

                "total_steps": self.total_steps,
                "mean_steps": self.mean_steps,

                "total_elapsed_time": self.total_elapsed_time,

                "mean_elapsed_time": self.mean_elapsed_time,
            },

            "action_noise": {
                "noisy_action_count": self.noisy_action_count,

                "total_recorded_actions": self.total_recorded_actions,

                "noisy_action_fraction": self.noisy_action_fraction,
            },

            "chase_onset": {
                "chase_started_count": self.chase_started_count,

                "mean_chase_start_safe_distance": self.mean_chase_start_safe_distance,

                "mean_chase_start_predator_distance": self.mean_chase_start_predator_distance,

                "safe_distance_observation_count": self.chase_start_safe_distance_count,

                "predator_distance_observation_count": self.chase_start_predator_distance_count,
            },
        }

    def raw_totals_dict(self) -> dict[str, Any]:
        """
        Return all raw accumulator fields.

        This is useful for debugging or future checkpoint support.
        """
        return asdict(self)