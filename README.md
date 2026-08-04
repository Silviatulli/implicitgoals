# Implicit Goals — SPR Extension

Research codebase for **Interactive Task Model Disambiguation using Information Theory** (Tulli et al., AAAI 2027 submission).

The project extends the implicit-subgoal querying framework of Tulli et al. (2026) by implementing and evaluating four principled query-reduction strategies (H1–H4) across five MDP families.

Code: https://github.com/Silviatulli/implicitgoals-spr

---

## Overview

When a robot is uncertain about a user's implicit goals, it must ask targeted questions to disambiguate between candidate task models. This codebase evaluates four strategies for selecting *which* bottleneck state to query next:

| Condition | Hypothesis | Description |
|---|---|---|
| **Str. VI** | Baseline | Optimal query cost via value iteration, uniform prior |
| **Info Gain** | H1 | Entropy-minimising query selection (Mirsky et al. 2018) |
| **Transition** | H2 | Bottleneck-dominance propagation — free negative inferences |
| **Proximity** | H3 | Goal-proximity softmax prior over hypotheses |
| **Frequency** | H4 | Greedy argmax of hypothesis overlap — lowest compute cost |

Environments: **Grid**, **Four Rooms**, **Puddle**, **Rock** (4×4 to 196×196), and **Overcooked** (38,417 states).

---

## Installation

```bash
git clone https://github.com/Silviatulli/implicitgoals-spr
cd implicitgoals
pip install -r requirements.txt
```

**Core dependencies**: `numpy`, `scipy`, `pandas`, `torch`, `networkx`, `matplotlib`

---

## Repository Structure

| File | Role |
|---|---|
| `parallel_experiments_2.py` | **Main experiment runner** — all five conditions, all environments |
| `overcooked_env.py` | Overcooked pipeline + simulation functions for all conditions |
| `plot_results.py` | Generates publication-quality LaTeX tables and figures |
| `QueryMDP.py` | Original query MDP formulation (Tulli et al. 2026) |
| `maximal_achievable_subsets.py` | Algorithm 1: maximal achievable subset enumeration |
| `experiments.py` | Grid/Puddle/Rock/Four Rooms world generators |
| `DeterminizedMDP.py` | MDP determinisation and bottleneck identification |
| `experiment_results_2/` | CSV results and run logs |
| `experiment_results/` | Figures (PDF/PNG) |

---

## How the Results Were Obtained

### Pipeline overview

Each trial follows this sequence:

```
1. Generate robot MDP (M_R) for the chosen environment and grid size
2. Assign each human model a random goal from 8 candidate positions
3. Build human MDPs (M_H) — one per unique goal (deduplicated)
4. Extract bottlenecks:
     - Small grids (≤400 states): dominator tree + toboggan filtering
     - Large grids (>400 states): fast-path — goal states used directly as bottlenecks
5. Build hypothesis space I_decoded (one hypothesis per unique human goal)
6. Solve Query MDP via vectorised backward induction → ExactQNet
7. For each human model simulate all five conditions and count oracle queries:
     - Str. VI     : ExactQNet policy lookup
     - Info Gain   : greedy one-step entropy minimisation
     - Transition  : ExactQNet + dominance propagation on negative responses
     - Proximity   : softmax prior from V*(M_R) → argmax P_τ(s ∈ IG)
     - Frequency   : argmax |{φ ∋ s}| over consistent hypotheses
     - Random      : uniform random over unqueried bottlenecks
8. Query All baseline = |B_filter| (ask about every bottleneck)
```

### Experiment parameters

| Parameter | Value |
|---|---|
| Grid sizes | 4, 10, 20, 50, 100, 196 (states: 16 → 38,416) |
| Human model counts \|H\| | 10, 50, 100 |
| Obstacle densities | 10%, 15%, 20% |
| Independent runs per config | 5 |
| Trials per run (grid) | 3 (macOS) / 5 (Linux) |
| Trials per run (Overcooked) | 3 seeds × 5-of-10 recipe subsets |
| Candidate human goals | 8 (corners + edge midpoints + centre) |
| Random seeds | `42 + run_idx` (deterministic) |

### Running the full sweep

```bash
python parallel_experiments_2.py
```

Runtime: ~70–120 minutes on a 4-core laptop (macOS).
Results are saved to `experiment_results_2/enhanced_pybullet_comparison.csv`.

### Regenerating tables and figures

```bash
python plot_results.py
```

Outputs:
- `experiment_results/query_counts_vs_models.pdf` — bar chart figure
- LaTeX for Tables 1, 2, 3 printed to stdout (copy into Overleaf)

---

## Key Implementation Details

### Condition implementations (`overcooked_env.py`)

**H1 — Info Gain** (`simulate_overcooked_info_gain`):  
At each step, scores every unqueried bottleneck by Shannon entropy reduction under a uniform prior and queries the maximiser.

**H2 — Transition** (`simulate_overcooked_transition`):  
Uses the same ExactQNet policy as Str. VI for *selection*, but after each negative oracle response propagates the dominance entailment `b2 ∉ IG ⟹ b1 ∉ IG` (where `∀φ: b1 ∈ φ ⟹ b2 ∈ φ`) to rule out dominated bottlenecks for free — without incrementing the query counter.

**H3 — Proximity** (`simulate_overcooked_proximity`):  
Computes `V*(M_R)` via reverse BFS from the robot's goal (`compute_v_star_grid`), assigns each hypothesis `ψ(φ) = mean V*(b)` over its bottlenecks, forms a softmax prior with τ=1, and queries the bottleneck with highest `P_τ(s ∈ IG)`.

**H4 — Frequency** (`simulate_overcooked_frequency`):  
At each step queries `argmax_s |{φ ∈ Φ(B, K_I) : s ∈ φ}|` — the bottleneck appearing in the most currently consistent hypotheses. No value function or entropy computation required.

### Fast pipeline (Yacine's contribution)

For small grids (≤400 states), bottlenecks are found via a **dominator tree** (NetworkX) followed by **toboggan filtering** to remove non-branching runs. The Query MDP is solved by **vectorised backward induction** over a base-3 encoded state space (3ⁿ states for n bottlenecks), yielding an `ExactQNet` — a lookup table with a PyTorch `nn.Module` interface for compatibility. This is *not* a neural network; it contains no learned weights.

For large grids (>400 states), corridor dominators are rare in open grids, so the dominator tree construction is skipped and goal states are used directly as bottlenecks (fast-path).

---

## Results Summary

| Condition | Overcooked queries | Grid (4×4) queries | Grid (≥20×20) |
|---|---|---|---|
| Str. VI (baseline) | 9.97 ± 1.41 | 3.9–4.2 | 3.9–4.1 |
| Info Gain (H1) | **4.63 ± 0.23** | 3.4–4.1 | converged |
| Transition (H2) | 8.99 ± 1.11 | ≈ Str. VI | converged |
| Proximity (H3) | **4.51 ± 0.37** | 3.5–4.0 | converged |
| Frequency (H4) | **4.51 ± 0.37** | 3.4–4.1 | converged |
| Random | 8.36 ± 0.89 | 3.9–4.1 | 3.9–4.0 |
| Query All | 17.00 ± 0.00 | 5.7–6.0 | 7.1–8.2 |

H1, H3, H4 reduce queries by **44–72%** vs Query All across all environments. All strategies converge at ≥400 states on grids. Transition (H2) provides minimal gains when hypotheses are non-overlapping.

---

## Reproducibility

All experiments use fixed seeds (`42 + run_idx`). Re-running `parallel_experiments_2.py` with the default parameters will reproduce the CSV within sampling noise from the obstacle/goal randomisation. For exact reproduction, fix the `random.randint` obstacle seeds in `generate_robot_model` and `generate_human_model_with_goal`.

---

## Citation

```
@inproceedings{tulli2027implicit,
  title     = {Interactive Task Model Disambiguation using Information Theory},
  author    = {Tulli, Silvia and Vasileiou, Stylianos Loukas and Sreedharan, Sarath
               and Meneguzzi, Felipe and Mirsky, Reuth},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```
