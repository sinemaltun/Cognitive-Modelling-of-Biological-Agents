from environment.grid_vi import GridVIEnvironment
from agents.value_iteration_agent import ValueIterationAgent

def run_training():
    env = GridVIEnvironment()
    env.reset()
    agent = ValueIterationAgent(gamma=0.9, theta=1e-4)

    agent.train(env)
    agent.save_model("models/vi_policy.pkl")

if __name__ == "__main__":
    run_training()