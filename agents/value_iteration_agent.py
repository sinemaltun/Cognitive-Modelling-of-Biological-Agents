import pickle
import random
from pathlib import Path
from typing import Callable, Iterable

from environment import Action
from planning import PlanningState, Transition


TransitionFunction = Callable[[PlanningState, Action],list[Transition],]


class ValueIterationAgent:
    def __init__(
        self,
        gamma: float = 0.95,
        theta: float = 1e-5,
        max_iterations: int = 1_000,
    ):
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")

        if theta <= 0:
            raise ValueError("theta must be positive.")

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.actions = list(Action)

        self.values: dict[PlanningState, float,] = {}

        self.policy: dict[PlanningState,Action,] = {}

    @property
    def policy_size(self) -> int:
        return len(self.policy)

    @property
    def value_function_size(self) -> int:
        return len(self.values)

    def metadata(self) -> dict:
        return {
            "gamma": self.gamma,
            "theta": self.theta,
            "max_iterations": (
                self.max_iterations
            ),
            "value_states": (
                self.value_function_size
            ),
            "policy_states": (
                self.policy_size
            ),
        }

    def train(self, states: Iterable[PlanningState],transition_function: TransitionFunction,) -> None:
        state_list = list(states)

        self.values = {state: 0.0 for state in state_list}

        for iteration in range(1, self.max_iterations + 1,):
            delta = 0.0
            next_values = {}

            for state in state_list:
                best_value = max(
                    self._action_value(
                        state,
                        action,
                        transition_function,
                    )
                    for action in self.actions
                )

                next_values[state] = best_value

                delta = max(delta, abs(best_value - self.values[state]),
                )

            self.values = next_values

            print(
                f"Iteration {iteration} | "
                f"maximum change={delta:.8f}"
            )

            if delta < self.theta:
                print(
                    "Value Iteration converged "
                    f"after {iteration} iterations."
                )
                break

        else:
            print(
                "Value Iteration reached the "
                "maximum iteration count."
            )

        self._extract_policy(state_list, transition_function)

    def choose_action(self, state: PlanningState,) -> Action:
        return self.policy.get(state, Action.STAY,)

    def _action_value(self, state: PlanningState, action: Action, transition_function: TransitionFunction,) -> float:
        expected_value = 0.0

        for transition in transition_function(state, action,):
            future_value = 0.0

            if not transition.terminal:
                future_value = self.values.get(transition.next_state, 0.0,)

            expected_value += (transition.probability * (transition.reward + self.gamma * future_value))

        return expected_value

    def _extract_policy(self, states: list[PlanningState], transition_function: TransitionFunction,) -> None:
        self.policy = {}

        for state in states:
            action_values = {
                action: self._action_value(state, action, transition_function,) for action in self.actions
            }

            best_value = max(action_values.values())

            best_actions = [action for action, value in action_values.items() if value == best_value]

            self.policy[state] = random.choice(best_actions)

    def save(self, path: str | Path,) -> None:
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "gamma": self.gamma,
                    "theta": self.theta,
                    "max_iterations": (
                        self.max_iterations
                    ),
                    "values": self.values,
                    "policy": self.policy,
                },
                file,
            )

    def load(self, path: str | Path,) -> None:
        with open(path, "rb") as file:
            data = pickle.load(file)

        self.gamma = data["gamma"]
        self.theta = data["theta"]

        self.max_iterations = data[
            "max_iterations"
        ]

        self.values = data["values"]
        self.policy = data["policy"]