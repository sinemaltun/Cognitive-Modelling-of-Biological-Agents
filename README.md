# Cognitive Modelling of Biological Agents

This project implements a grid-based approach–avoidance task in which an agent collects tokens while managing the risk of a predator becoming active. The environment supports human play, random behavior, temporal-difference learning with SARSA and Q-learning, and model-based planning with Value Iteration.

The current game uses a 24 × 16 grid with a player, collectible tokens, a predator, and a safe zone. Each trial begins in a foraging phase. If a threat event occurs, the game enters a chase phase and the agent must avoid capture or reach safety.

## Project structure

```text
moba-v1.1/
├── agents/
├── config/
├── entities/
├── environment/
├── experiments/
├── models/
├── planning/
└── visualization/
```

## `agents/`

Contains all decision-making agents and shared learning infrastructure.

- **`base_agent.py`** — Defines the abstract `BaseAgent` interface. Agents implement `choose_action()`, while learning agents may also implement `update()`.
- **`human_agent.py`** — Converts Pygame keyboard events into environment actions. Arrow keys move the player, Space selects `STAY`, and Escape requests exit.
- **`random_agent.py`** — Selects a uniformly random action. It is useful as a basic behavioral baseline and for testing the environment.
- **`tabular_td_agent.py`** — Shared implementation for tabular temporal-difference agents. It owns the Q-table, epsilon-greedy action selection, random tie-breaking, epsilon decay, action indexing, and model serialization.
- **`sarsa_agent.py`** — Implements the on-policy SARSA update. Its target uses the value of the next action actually chosen by the current policy.
- **`qlearning_agent.py`** — Implements the off-policy Q-learning update. Its target uses the highest Q-value available in the next state.
- **`value_iteration_agent.py`** — Implements Value Iteration over an explicit transition model. It repeatedly updates state values until convergence and then extracts a policy.
- **`phase_aware_value_iteration_agent.py`** — Adapts the two Value Iteration policies to the live game. It uses a nearest-token policy during foraging and a player–predator policy during chase.
- **`__init__.py`** — Re-exports the agent classes for convenient imports such as `from agents import SARSAAgent`.

## `config/`

Contains centralized experiment configuration.

- **`rewards.py`** — Defines the default reward and reward-shaping values used by the live environment and planning model. These include step cost, token reward, escape and survival rewards, movement shaping, and the gradual penalty for foraging far from the safe zone.
- **`__init__.py`** — Marks the directory as a Python package.

The live environment copies the default reward dictionary when it is created, so individual experiments can override selected values without changing the global defaults.

## `entities/`

Contains small data-oriented classes representing objects in the game.

- **`position.py`** — Defines the immutable `Position` dataclass and provides movement, tuple conversion, and Manhattan-distance calculations.
- **`player.py`** — Stores the player position and collected-token count. It provides methods for movement, token collection, and token loss.
- **`predator.py`** — Stores predator position, wake state, and speed. It implements deterministic Manhattan pursuit and collision detection.
- **`token.py`** — Defines an immutable token with a grid position.
- **`safezone.py`** — Defines the safe-zone position and checks whether another position lies inside it.
- **`__init__.py`** — Re-exports all entity types.

## `environment/`

Contains the live game environment used by humans and learning agents.

- **`game.py`** — The central environment implementation. It defines:
  - the `Phase`, `Status`, and `Action` enums;
  - the foraging, chase, and terminal phase transitions;
  - real-time and accelerated simulated-time modes;
  - player movement, token collection and respawning;
  - predator wake-up, pursuit, and capture;
  - reward calculation and shaping;
  - the Gym-like `reset()` and `step()` interface.
- **`grid.py`** — Defines the grid dimensions and boundary check used for movement.
- **`state_features.py`** — Extracts relative spatial features from the live game. `StateFeatures` provides distances to the nearest token, predator, and safe zone. `build_td_state()` converts these features into the hashable tuple used by SARSA and Q-learning.
- **`__init__.py`** — Re-exports the primary environment classes, enums, and state-building functions.

The public interaction pattern is:

```python
state = env.reset()
next_state, reward, done, info = env.step(action)
```

For human play and animated demonstrations, use `realtime=True`. For fast TD training, use `realtime=False`; each call to `step()` then advances simulated time by `1 / steps_per_second` seconds.

## `planning/`

Contains the explicit environment model required by Value Iteration. This is deliberately separate from the mutable live environment.

- **`states.py`** — Defines immutable planning states:
  - `ForagePlanningState` stores the exact player and current target-token coordinates;
  - `ChasePlanningState` stores the exact player and predator coordinates.
- **`model.py`** — Defines `GridPlanningModel` and `Transition`. The model enumerates possible action outcomes, computes rewards and terminal conditions, simulates player and predator movement, and generates the full state sets used by Value Iteration.
- **`__init__.py`** — Re-exports planning states, transitions, and the model.

The Value Iteration abstraction is intentionally smaller than the complete live game state. During foraging it plans toward one target token at a time; during chase it plans from exact player and predator positions. This avoids enumerating every possible arrangement of all tokens, timers, and accumulated rewards.

## `visualization/`

Contains the Pygame visualization layer.

- **`pygame_renderer.py`** — Draws the grid, player, tokens, predator, safe zone, phase-colored frame, timer, status, and token counter. It reads environment state but does not implement game rules.
- **`__init__.py`** — Re-exports `PygameRenderer`.

The renderer follows the experiment-inspired visual language: a green triangular player, yellow diamond-shaped tokens, a gray or red predator, a black safe zone, and a blue or red frame depending on the phase.

## `experiments/`

Contains executable scripts that assemble environments, agents, models, and renderers.

### Interactive and baseline runs

- **`run_human.py`** — Runs the real-time environment with keyboard control.
- **`run_random.py`** — Runs and displays the random-agent baseline.

### TD training and playback

- **`train_sarsa.py`** — Trains SARSA in accelerated simulated time and saves its Q-table to `models/sarsa_agent.pkl`.
- **`run_trained_sarsa.py`** — Loads the saved SARSA Q-table and displays its greedy policy in real time.
- **`train_qlearning.py`** — Trains Q-learning in accelerated simulated time and saves its Q-table to `models/qlearning_agent.pkl`.
- **`run_trained_qlearning.py`** — Loads and visualizes the greedy Q-learning policy.

### Value Iteration planning and playback

- **`train_value_iteration.py`** — Builds the explicit planning model, trains separate foraging and chase policies, and saves serial-numbered model pairs.
- **`run_trained_value_iteration.py`** — Loads a selected foraging/chase model pair and runs the phase-aware Value Iteration agent in the live environment.
- **`__init__.py`** — Marks the directory as a package so scripts can be run with Python's `-m` option.

Run scripts from the project root, for example:

```bash
python -m experiments.run_human
python -m experiments.train_sarsa
python -m experiments.run_trained_sarsa
python -m experiments.train_qlearning
python -m experiments.run_trained_qlearning
python -m experiments.train_value_iteration
python -m experiments.run_trained_value_iteration
```

## `models/`

Stores serialized learned Q-tables and planned policies.

Current examples include:

- `sarsa_agent.pkl`
- `value_iteration_forage_001.pkl`
- `value_iteration_chase_001.pkl`

Model files are Python pickle files. Only load pickle files from trusted sources. Q-learning playback expects `qlearning_agent.pkl`, which is created after running its training script.

## Learning approaches

### SARSA

SARSA is an on-policy temporal-difference method. It learns from sampled interaction and updates the value of the current state/action pair using the next action actually chosen by its epsilon-greedy policy. Exploration is therefore reflected in the learned values.

### Q-learning

Q-learning is an off-policy temporal-difference method. It also learns from sampled interaction, but its update assumes the best-valued action will be selected in the next state, independently of the exploratory action actually taken.

### Value Iteration

Value Iteration is model-based. Instead of learning transition consequences only through sampled episodes, it uses `GridPlanningModel` to evaluate all modeled outcomes for every state/action pair. Separate policies are computed for foraging and chase and joined by `PhaseAwareValueIterationAgent` at runtime.

## Installation

Python 3.10 or newer is recommended because the project uses modern union type syntax such as `str | Path`.

Install the external dependencies:

```bash
python -m pip install numpy pygame
```

Then run commands from the directory containing `agents/`, `environment/`, and the other project packages.

## Controls

During human play:

- Arrow keys: move up, right, down, or left
- Space: stay in place
- Escape or window close button: exit

## Notes and current limitations

- The live game and the Value Iteration planning model use related but not completely identical state representations.
- Value Iteration's foraging policy targets one nearest token at a time rather than modeling all token placements simultaneously.
- The chase planning model uses a fixed capture penalty, whereas the live game removes the reward value of collected tokens when capture occurs.
- Value Iteration over the full 24 × 16 position combinations is computationally more expensive than running a single TD episode.
- Saved models depend on the state representation, action order, reward design, and environment dynamics used during training. Retrain models after changing those elements.
