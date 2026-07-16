import numpy as np

from agents.tabular_td_agent import TabularTDAgent


class QLearningAgent(TabularTDAgent):
    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        action_index = self.action_index(action)

        current_q = self.q[state][action_index]

        if done:
            target = reward
        else:
            target = (
                reward
                + self.gamma
                * np.max(self.q[next_state])
            )

        self.q[state][action_index] += (
            self.alpha * (target - current_q)
        )