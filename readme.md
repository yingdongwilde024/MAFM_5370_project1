# Continuous Asset Allocation with RL (PPO)
A reinforcement learning environment for continuous-time asset allocation using PPO algorithm, supporting risky assets + risk-free cash with realistic constraints.

## Environment Requirements
- Python 3.8+
- Dependencies:
  ```bash
  pip install gymnasium stable-baselines3 numpy matplotlib
  ```

## Core Parameters
| Parameter          | Description                                                                 | Default Value          | Constraints       |
|--------------------|-----------------------------------------------------------------------------|------------------------|-------------------|
| `n`                | Number of risky assets                                                      | 3                      | n < 5             |
| `T`                | Number of investment periods                                                | 8                      | T < 10            |
| `r`                | Risk-free cash interest rate                                                | 0.02 (2%)              | -                 |
| `a`                | Expected returns of risky assets                                            | [0.05, 0.08, 0.12, ...]| Length = n        |
| `s`                | Variances of risky asset returns                                            | [0.01, 0.04, 0.09, ...]| Length = n        |
| `p_init`           | Initial portfolio weights (cash + risky assets)                             | [0.4, 0.2, 0.2, 0.2]   | Sum = 1           |
| `aversion_rate`    | CARA risk aversion coefficient (λ)                                          | 1.0                    | Positive float    |