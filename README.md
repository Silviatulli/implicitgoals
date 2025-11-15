# Implicit Goals

A research codebase for analyzing implicit goals and bottlenecks in Markov Decision Processes (MDPs). Implements algorithms to identify and compare achievable goal subsets between robot and human models in various grid-based and continuous environments.

## Installation

1. Clone the repository and navigate to the directory:
```bash
git clone <repository-url>
cd implicit_goals_mdp
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Dependencies**: numpy, scipy, pandas, gymnasium, mani-skill2 (optional), sapien (optional), colorama

## Project Structure

- **MDP Environments**: `GridWorldClass.py`, `PuddleWorldClass.py`, `RockWorldClass.py`, `TaxiWorldClass.py`, `MinigridWorldClass.py`, `ManiskillClass.py`
- **Core Algorithms**: `MDP.py`, `Utils.py`, `DeterminizedMDP.py`, `BottleneckCheckMDP.py`, `QueryMDP.py`, `Search.py`
- **Analysis Tools**: `maximal_achievable_subsets.py`, `parallel_experiments.py`, `minigrid_tests.py`

## Key Concepts

- **Bottlenecks**: Critical states that must be traversed to reach goal states, identified by analyzing determinized MDPs
- **Maximal Achievable Subsets**: Largest subsets of bottleneck states that a robot model can simultaneously achieve when compared to human models
- **Query MDP**: Constructs a new MDP where states represent combinations of achieved bottlenecks

## Usage Examples

### Basic Grid World

```python
from GridWorldClass import GridWorld
from Utils import ValueIteration, get_policy

grid_world = GridWorld(size=10, start=(0, 0), goal=(9, 9), obstacles_percent=0.15)
V = ValueIteration(grid_world)
policy = get_policy(grid_world, V)
```

### Bottleneck Identification

```python
from DeterminizedMDP import DeterminizedMDP, identify_bottlenecks

det_mdp = DeterminizedMDP(env)
bottlenecks = identify_bottlenecks(det_mdp)
```

### Maximal Achievable Subsets

```python
from maximal_achievable_subsets import find_maximally_achievable_subsets
from GridWorldClass import generate_and_visualize_gridworld

M_R = generate_and_visualize_gridworld(size=10, start=(0,0), goal=(9,9), obstacles_percent=0.1)
M_H_list = [generate_and_visualize_gridworld(size=10, start=(0,0), goal=(9,9), 
              obstacles_percent=0.15, obstacle_seed=i) for i in range(1, 6)]

I, B = find_maximally_achievable_subsets(M_R, M_H_list)
```

### Parallel Experiments

```python
from parallel_experiments import run_parallel_experiments_with_obstacles

results = run_parallel_experiments_with_obstacles(
    grid_sizes=[10, 15, 20],
    num_models=[3, 5],
    world_types=['grid', 'puddle', 'rock'],
    obstacle_percents=[0.1, 0.15, 0.2],
    num_trials=5,
    max_workers=4
)
```

## Configuration

**Random Seeds**: Each experiment configuration gets a unique seed from range 1-10000. Robot and human models use the same seed as their configuration.

**Default Parameters** (edit in `parallel_experiments.py`):
- `num_trials = 5` per configuration
- `max_workers = 4` for parallel processing
- `batch_size = 10` for batch processing

**Results**: Saved to `experiment_results/` directory as CSV files. Logs written to `experiment.log`.

## Customization

Environments can be customized by modifying their respective class files:
- **GridWorld**: `size`, `obstacles_percent`, `obstacle_seed`, `divide_rooms`, `slip_prob`
- **PuddleWorld**: `puddle_percent`, `puddle_penalty` (inherits from GridWorld)
- **RockWorld**: `rock_percent`, rock interaction mechanics

## Performance & Troubleshooting

**Memory Issues**: Use sparse value iteration (`sparse_value_iteration` in `Utils.py`), reduce grid size, or increase epsilon threshold.
**Large State Spaces**: Sparse implementations are recommended. Parallel processing available for batch experiments.
**Path Finding Failures**: Increase `max_tries` parameter, reduce obstacle percentage, or adjust start/goal positions.
**ManiSkill2**: Requires GPU drivers and CUDA. See ManiSkill2 documentation for setup.

## Running Tests

```bash
python minigrid_tests.py
```
