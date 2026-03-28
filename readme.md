# Discrete Asset Allocation RL Environment

This repository implements a **discrete-time asset allocation environment** for reinforcement learning (RL), built on OpenAI Gymnasium. The environment simulates portfolio optimization over a finite horizon with multiple risky assets + a risk-free cash asset, and uses Proximal Policy Optimization (PPO) to train an agent to maximize CARA (Constant Absolute Risk Aversion) utility of terminal wealth.

## Key Features
- **Environment Constraints**: Configurable number of risky assets (2 < n < 5) and time horizon (T < 10), aligned with practical portfolio optimization scenarios.
- **CARA Utility Reward**: Dense, potential-based reward shaping (incremental CARA utility) that preserves the optimal policy (cumulative reward = U(W_T) - U(W₀)).
- **Turnover Constraint**: 10% maximum portfolio turnover per step to reflect real-world trading costs/liquidity limits.
- **VecNormalize**: Observation/reward normalization for stable RL training.
- **Generalizability**: Supports custom asset return parameters (mean/var), initial portfolio weights, risk-free rate, and risk aversion.

## Installation

### Dependencies
Install required packages via `pip`:
```bash
pip install gymnasium numpy stable-baselines3 matplotlib
```

### Clone Repository
```bash
git clone <your-repo-url>
cd <repo-name>
```

## Environment Details

### `DiscreteAssetAllocationEnv`
Core environment class implementing the asset allocation problem:
- **State (Observation)**: `[t/T, W, p₀, p₁, ..., pₙ]`  
  - `t/T`: Normalized time step (0 to 1)
  - `W`: Current portfolio wealth (starts at 1.0)
  - `p₀...pₙ`: Portfolio weights (cash + risky assets, sum to 1)
- **Action**: `n+1` logits → softmax to enforce non-negative weights summing to 1 (no short-selling).
- **Turnover Control**: Scales portfolio adjustments to respect 10% maximum turnover per step.
- **Market Dynamics**: Risky asset returns follow normal distribution `N(a(k), s(k))` (mean `a(k)`, variance `s(k)`); cash returns are fixed at risk-free rate `r`.
- **Reward**: Incremental CARA utility: `U(W_t) - U(W_{t-1})`, where `U(W) = -exp(-λ·W)/λ` (λ = risk aversion rate).

### Key Parameters
| Parameter         | Description                                  | Default                                  |
|-------------------|----------------------------------------------|------------------------------------------|
| `n`               | Number of risky assets (3 ≤ n ≤ 4)           | 3                                        |
| `T`               | Time horizon (≤9)                            | 8                                        |
| `r`               | Risk-free rate (cash return)                 | 0.02                                     |
| `a`               | Mean returns of risky assets                 | `[0.05, 0.08, 0.11, ...]` (per n)        |
| `s`               | Variances of risky asset returns             | `[(0.1)², (0.2)², (0.3)², ...]` (per n)  |
| `p_init`          | Initial portfolio weights (cash + risky)     | `[0.4, 0.6/n, ...]` (sum to 1)           |
| `aversion_rate`   | CARA risk aversion parameter (λ)             | 1.0                                      |

## Usage

### Run the Full Pipeline
The main script trains a PPO agent on 4 representative scenarios and generates plots of wealth/portfolio weights:
```bash
python project1.py
```

### Scenarios Included
1. **Baseline**: n=3, T=5, default parameters
2. **Max Assets**: n=4, T=8, default parameters
3. **High Risk Aversion**: n=3, T=9, high risk-free rate (0.05), λ=2.0
4. **Custom Parameters**: n=4, T=6, custom mean/var/initial weights, λ=1.5

### Custom Scenarios
To define your own scenario, add a dictionary to the `scenarios` list in the main block:
```python
{
    "name": "Custom Scenario",
    "n": 3, "T": 7,
    "r": 0.03,
    "a": [0.07, 0.09, 0.12],  # Mean returns for 3 risky assets
    "s": [0.008, 0.015, 0.02], # Variances
    "p_init": [0.3, 0.2, 0.3, 0.2], # Initial weights (cash + 3 assets)
    "aversion_rate": 1.2,
}
```

## Outputs
1. **Console Logs**: Step-by-step wealth and portfolio weights for each test episode.
2. **Plot File**: `results.png` (saved to working directory) with:
   - Wealth trajectory (W(t)) over time for each scenario
   - Portfolio weight dynamics (cash + each risky asset)

## Training Details
- **RL Algorithm**: PPO (Proximal Policy Optimization) from Stable Baselines3
- **Learning Rate**: Linear schedule (starts at 0.001, decays to 0)
- **Normalization**: `VecNormalize` (obs/reward normalization, clip obs to ±10)
- **Policy**: MLP (Multi-Layer Perceptron) with weight decay (1e-4)
- **Training Steps**: 300,000 timesteps per scenario (configurable via `total_timesteps`)

## Key Functions
| Function          | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `linear_schedule` | Linear learning rate decay (progress-based)                             |
| `run_scenario`    | End-to-end training + test pipeline for a single scenario               |
| `reset`/`step`    | Gym environment core methods (state reset, step execution)             |

## Notes
- **Deterministic Testing**: After training, the agent is evaluated with deterministic actions to ensure reproducibility.
- **Temp Files**: Model/stats are saved to temporary directories during evaluation (no persistent files).
- **Wealth Drift**: Portfolio weights automatically adjust ("drift") with asset returns to reflect passive price changes.

## License
This project is for research/educational purposes. Feel free to modify and extend for your own use cases.

## References
- CARA Utility: [Constant Absolute Risk Aversion](https://en.wikipedia.org/wiki/Risk_aversion#Constant_absolute_risk_aversion)
- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Gymnasium: [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- Stable Baselines3: [SB3 Documentation](https://stable-baselines3.readthedocs.io/)
