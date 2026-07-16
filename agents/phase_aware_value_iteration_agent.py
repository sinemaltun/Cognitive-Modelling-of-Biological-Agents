from environment import Action, Phase

from agents.value_iteration_agent import (
    ValueIterationAgent,
)

from planning import (
    ChasePlanningState,
    ForagePlanningState,
)


class PhaseAwareValueIterationAgent:
    """
    Adapter that connects the two Value Iteration policies
    to the live ForagingGame.
    """

    def __init__(
        self,
        forage_agent: ValueIterationAgent,
        chase_agent: ValueIterationAgent,
    ):
        self.forage_agent = forage_agent
        self.chase_agent = chase_agent

    def choose_action(self, env) -> Action:
        if env.phase == Phase.FORAGING:
            nearest_token = min(
                env.tokens,
                key=lambda token:
                    env.player.position.manhattan_distance(token.position),
            )

            planning_state = (
                ForagePlanningState(
                    player_x=env.player.position.x,
                    player_y=env.player.position.y,
                    token_x=nearest_token.position.x,
                    token_y=nearest_token.position.y,
                )
            )

            return self.forage_agent.choose_action(planning_state)

        if env.phase == Phase.CHASE:
            planning_state = ChasePlanningState(
                player_x=env.player.position.x,
                player_y=env.player.position.y,
                predator_x=env.predator.position.x,
                predator_y=env.predator.position.y,
            )

            return self.chase_agent.choose_action(planning_state)

        return Action.STAY