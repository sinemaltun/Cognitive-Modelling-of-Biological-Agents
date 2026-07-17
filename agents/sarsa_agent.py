from agents.tabular_td_agent import TabularTDAgent


class SARSAAgent(TabularTDAgent):
    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done,
    ):
        action_index = self.action_index(action)

        current_q = self.q[state][action_index]

        if done:
            target = reward
        else:
            next_action_index = self.action_index(next_action)

            target = (reward+ self.gamma* self.q[next_state][next_action_index])

        self.q[state][action_index] += (self.alpha * (target - current_q))