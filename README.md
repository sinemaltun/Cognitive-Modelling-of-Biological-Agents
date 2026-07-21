# Cognitive Modelling of Biological Agents

A reinforcement learning framework for modelling biologically-inspired decision making under **approach–avoidance conflict**.

This project was developed for the **Cognitive Modelling of Biological Agents** course at the **University of Bonn**. It recreates a grid-based foraging task inspired by the experimental paradigm of Bach *et al.*, in which an agent must balance reward-seeking behaviour against the risk of predation. The framework supports both reinforcement learning agents and human participants, enabling behavioural comparisons under identical experimental conditions.

---

# Motivation

Many real-world decisions require balancing potential reward against potential danger. This **approach–avoidance conflict** is believed to involve neural mechanisms such as the hippocampus and plays an important role in human decision making.

To investigate this computationally, the project recreates a simplified foraging task in which an agent

- collects reward tokens,
- avoids an awakening predator,
- decides when continued foraging becomes too risky,
- attempts to safely reach a refuge once threatened.

Rather than evaluating agents solely by cumulative reward, the framework also records behavioural measures inspired by cognitive modelling.

---

# Features

- 24 × 16 grid-world environment
- Two-phase gameplay (Foraging and Chase)
- Human-controlled baseline
- Random baseline
- SARSA
- Q-Learning
- Value Iteration
- Phase-aware Value Iteration
- Configurable predator probability
- Optional action noise
- Real-time and accelerated simulation
- Behavioural logging
- Unified CSV / JSON evaluation pipeline
- Interactive visualization using Pygame

---

# Project Architecture

```
Experiment
      │
      ▼
   Agent ───────────────┐
      │                 │
      ▼                 │
 Environment            │
      │                 │
      ▼                 │
Visualization           │
      │                 │
      ▼                 │
 Evaluation ◄───────────┘
```

---

# Repository Structure

```text
moba/
│
├── agents/                # Reinforcement learning algorithms
├── config/                # Reward configuration
├── entities/              # Core game objects
├── environment/           # Grid-world environment
├── evaluation/            # Logging and evaluation
├── experiments/           # Executable experiment scripts
├── planning/              # Value Iteration model
├── visualization/         # Pygame renderer
│
├── models/                # Trained agents (created automatically)
├── results/               # Experiment outputs (created automatically)
│
├── requirements.txt
└── README.md
```

The implementation packages form the framework itself. The `models/` and `results/` directories are created automatically when experiments are executed.

---

# Package Overview

## `agents/`

Implements every decision-making algorithm used by the framework.

Included agents:

- HumanAgent
- RandomAgent
- SARSAAgent
- QLearningAgent
- ValueIterationAgent
- PhaseAwareValueIterationAgent

Responsibilities:

- policy execution
- temporal-difference learning
- value updates
- exploration strategies
- action selection

---

## `environment/`

Implements the complete grid-world simulation.

Responsibilities include

- player movement
- predator behaviour
- reward computation
- phase transitions
- episode termination
- state representation

The environment exposes a Gym-like interface:

```python
state = env.reset()
next_state, reward, done, info = env.step(action)
```

---

## `planning/`

Contains the explicit planning model used by Value Iteration.

Responsibilities include

- planning state definitions
- transition model
- reward model
- Value Iteration
- policy extraction

Unlike the live environment, the planning model separates the problem into independent **Foraging** and **Chase** state spaces, making Value Iteration computationally feasible.

---

## `evaluation/`

Provides a unified evaluation and logging framework.

Outputs include

- step-by-step behaviour logs
- episode summaries
- training progress
- Value Iteration convergence
- run configuration
- run summaries

Every experiment uses the same logging infrastructure.

---

## `visualization/`

Renders the environment using Pygame.

The renderer is intended for

- demonstrations
- debugging
- observing learned behaviour
- human-controlled experiments

---

## `experiments/`

Contains executable entry points.

Typical scripts include

- training reinforcement learning agents
- evaluating trained agents
- replaying trained agents
- running human participants

---

# Learning Approaches

## SARSA

SARSA is an **on-policy** temporal-difference algorithm.

The agent updates its Q-values using the action actually selected by its current ε-greedy policy.

---

## Q-Learning

Q-Learning is an **off-policy** temporal-difference algorithm.

Updates assume the best available action will be taken in the next state, independent of exploration.

---

## Value Iteration

Value Iteration is a **model-based** planning algorithm.

Rather than learning through sampled episodes, it uses an explicit transition model to compute an optimal policy over the planning state space.

Separate policies are learned for

- Foraging
- Chase

and combined during gameplay by the `PhaseAwareValueIterationAgent`.

---

# Evaluation Framework

The framework records both traditional reinforcement learning metrics and behavioural statistics.

Examples include

- cumulative reward
- collected tokens
- survival rate
- threat survival rate
- successful escapes
- episode duration
- noisy actions
- chase onset statistics
- movement behaviour

Results are exported as CSV files together with machine-readable JSON summaries.

---

# Generated Outputs

Running experiments automatically creates

```text
models/
results/
```

## `models/`

Contains trained reinforcement learning models.

Examples:

```text
sarsa_training_<timestamp>.pkl
qlearning_training_<timestamp>.pkl

value_iteration_training_<timestamp>_forage.pkl
value_iteration_training_<timestamp>_chase.pkl
```

---

## `results/`

Each experiment creates its own timestamped directory.

Typical contents:

```text
run_config.json
run_summary.json
episodes.csv
steps.csv
training_progress.csv
value_iteration_progress.csv
```

depending on the experiment.

---

# Running the Project

## Installation

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

or

```bash
python -m pip install numpy pygame
```

---

## Important

Run all experiment scripts **from the project root directory**.

Example:

```bash
cd path/to/moba
```

Run scripts as Python modules:

```bash
python -m experiments.<script_name>
```

instead of running the script file directly:

```bash
python experiments/<script_name>.py
```

The module-based form works on **Windows, Linux and macOS** and ensures that all project packages are imported correctly.

---

# Human Experiments

Run a human-controlled experiment while recording the same behavioural metrics used for the reinforcement learning agents.

```bash
python -m experiments.run_human
```

Optional arguments include:

```text
--participant-id
--episodes
--threat-probability
--action-noise
```

---

# Training

Training creates a new agent from scratch, learns (or computes) its policy, and saves the resulting model together with training statistics.

The generated model filenames contain a timestamp indicating when the training run was performed. Throughout this README, `<timestamp>` denotes this value (for example, `20260721_173106`).

## SARSA

```bash
python -m experiments.train_sarsa
```

Creates

```text
models/
    sarsa_training_<timestamp>.pkl

results/
    sarsa_training_<timestamp>/
```

---

## Q-Learning

```bash
python -m experiments.train_qlearning
```

Creates

```text
models/
    qlearning_training_<timestamp>.pkl

results/
    qlearning_training_<timestamp>/
```

---

## Value Iteration

```bash
python -m experiments.train_value_iteration
```

Creates

```text
models/
    value_iteration_training_<timestamp>_forage.pkl
    value_iteration_training_<timestamp>_chase.pkl

results/
    value_iteration_training_<timestamp>/
```

---

# Running Trained Agents

These scripts load a previously trained model and visualise its behaviour in the Pygame environment. No further learning takes place during these runs.

The playback scripts currently use model paths defined directly inside each script. Before running one of the scripts below, update the corresponding model path(s) so that they point to an existing file in the `models/` directory.

## SARSA

Open

```text
experiments/run_trained_sarsa.py
```

and set the model path, for example:

```python
MODEL_PATH = MODEL_DIR / "sarsa_training_20260721_173106.pkl"
```

Then run:

```bash
python -m experiments.run_trained_sarsa
```

---

## Q-Learning

Open

```text
experiments/run_trained_qlearning.py
```

and set the model path, for example:

```python
MODEL_PATH = MODEL_DIR / "qlearning_training_20260721_173106.pkl"
```

Then run:

```bash
python -m experiments.run_trained_qlearning
```

---

## Value Iteration

Open

```text
experiments/run_trained_value_iteration.py
```

and set both model paths, for example:

```python
FORAGE_MODEL_PATH = (
    MODEL_DIR
    / "value_iteration_training_20260721_173106_forage.pkl"
)

CHASE_MODEL_PATH = (
    MODEL_DIR
    / "value_iteration_training_20260721_173106_chase.pkl"
)
```

Both files should come from the same Value Iteration training run.

Then run:

```bash
python -m experiments.run_trained_value_iteration
```

---

# Evaluation

Evaluation loads a trained model, disables further learning and exploration, executes a fixed number of evaluation episodes, and records behavioural statistics.

Replace `<timestamp>` with the timestamp of the model you wish to evaluate (for example, `20260721_173106`).

Each evaluation script supports additional command-line arguments. Use `--help` to display all available options.

## SARSA

Display the available options:

```bash
python -m experiments.evaluate_sarsa --help
```

Evaluate a trained SARSA model:

```bash
python -m experiments.evaluate_sarsa models/sarsa_training_<timestamp>.pkl
```

Example:

```bash
python -m experiments.evaluate_sarsa models/sarsa_training_20260721_173106.pkl
```

---

## Q-Learning

Display the available options:

```bash
python -m experiments.evaluate_qlearning --help
```

Evaluate a trained Q-Learning model:

```bash
python -m experiments.evaluate_qlearning models/qlearning_training_<timestamp>.pkl
```

Example:

```bash
python -m experiments.evaluate_qlearning models/qlearning_training_20260721_173106.pkl
```

---

## Value Iteration

Display the available options:

```bash
python -m experiments.evaluate_value_iteration --help
```

Evaluate a trained Value Iteration agent by providing both the foraging and chase models from the same training run:

```bash
python -m experiments.evaluate_value_iteration models/value_iteration_training_<timestamp>_forage.pkl models/value_iteration_training_<timestamp>_chase.pkl
```

Example:

```bash
python -m experiments.evaluate_value_iteration models/value_iteration_training_20260721_173106_forage.pkl models/value_iteration_training_20260721_173106_chase.pkl
```

The commands above use forward slashes because they work on Windows, Linux and macOS. On Windows, backslashes may also be used if preferred.

Each evaluation creates a new timestamped directory inside

```text
results/
```

containing

```text
run_config.json
run_summary.json
episodes.csv
steps.csv
```

---

# Typical Workflow

```
Train Agent
      │
      ▼
Saved Model
      │
      ├────────► Run Trained Agent
      │               │
      │               ▼
      │        Observe Behaviour
      │
      └────────► Evaluate Agent
                      │
                      ▼
             CSV / JSON Results
```

---

# Future Extensions

The framework is intentionally modular and can be extended with

- Deep Reinforcement Learning
- Partial Observability
- Noisy Perception Models
- Human Behavioural Model Fitting
- Alternative reward structures
- Additional biologically-inspired decision models

---

# Authors

**Adam Krupinski**

**Sinem Altun**

University of Bonn