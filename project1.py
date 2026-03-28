import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from typing import Callable
import os
import tempfile


class DiscreteAssetAllocationEnv(gym.Env):
    """
    Discrete-time Asset Allocation Environment (n risky assets + 1 risk-free cash asset)
    Constraints: 2 < n < 5, T < 10

    Reward: Dense incremental CARA utility (potential-based reward shaping).
    Cumulative episodic reward = U(W_T) - U(W_0), preserving optimal policy.
    """

    def __init__(self, n=3, T=8, r=0.02, a=None, s=None, p_init=None, aversion_rate=1.0):
        super().__init__()

        assert n > 2, f"Constraint violation: n must be > 2, got {n}"
        assert n < 5, f"Constraint violation: n must be < 5, got {n}"
        assert T < 10, f"Constraint violation: T must be < 10, got {T}"

        self.n = n
        self.T = T
        self.r = r
        self.aversion_rate = aversion_rate

        # Mean returns a(k)
        if a is not None:
            self.a = np.array(a, dtype=np.float64)
            assert len(self.a) == self.n, "Dimension of 'a' must match 'n'"
        else:
            self.a = np.array([0.05 + i * 0.03 for i in range(self.n)])

        # Return variances s(k)
        if s is not None:
            self.s = np.array(s, dtype=np.float64)
            assert len(self.s) == self.n, "Dimension of 's' must match 'n'"
        else:
            self.s = np.array([(0.1 * (i + 1)) ** 2 for i in range(self.n)])

        # Initial portfolio weights p(k), index 0 = cash
        if p_init is not None:
            self.p_init = np.array(p_init, dtype=np.float64)
            assert len(self.p_init) == self.n + 1, "Dimension of 'p_init' must be n + 1"
            assert np.isclose(np.sum(self.p_init), 1.0), "Initial portfolio weights must sum to 1"
        else:
            self.p_init = np.array([0.4] + [0.6 / n] * n)

        # Initial CARA utility U(W_0) = -exp(-lambda * 1) / lambda, used as reset potential
        self.u_W0 = -np.exp(-self.aversion_rate * 1.0) / self.aversion_rate

        # Observation: [t/T, W, p_0, ..., p_n]
        # W is theoretically unbounded in both directions (normal returns, no floor on wealth)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n + 3,), dtype=np.float32
        )

        # Action: n+1 logits -> softmax -> portfolio weights (non-negative, sum to 1)
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.n + 1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.W = 1.0  # X(0) = 1
        self.p = self.p_init.copy()
        self.prev_utility = self.u_W0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.concatenate(([self.t / self.T, self.W], self.p))
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Softmax: no short-selling, weights sum to 1
        exp_a = np.exp(action - np.max(action))
        p_target = exp_a / np.sum(exp_a)

        # 2. 10% turnover constraint: scale adjustment if needed
        diff = p_target - self.p
        turnover = 0.5 * np.sum(np.abs(diff))
        alpha = min(1.0, 0.1 / turnover) if turnover > 0.1 else 1.0
        p_next = self.p + alpha * diff

        # 3. Market evolution: one-period return of asset k ~ N(a(k), s(k))
        risky_returns = self.np_random.normal(self.a, np.sqrt(self.s))
        all_returns = np.concatenate(([self.r], risky_returns))

        port_return = np.sum(p_next * all_returns)
        self.W = self.W * (1 + port_return)

        # Passive weight drift due to price changes
        self.p = p_next * (1 + all_returns) / (1 + port_return)

        self.t += 1
        terminated = bool(self.t >= self.T)

        # 4. Dense reward: incremental CARA utility (potential-based shaping)
        current_utility = -np.exp(-self.aversion_rate * self.W) / self.aversion_rate
        reward = current_utility - self.prev_utility
        self.prev_utility = current_utility

        # === 将真实的终止状态放入 info 中 ===
        info = {
            "terminal_W": self.W,
            "terminal_p": self.p.copy()
        }

        return self._get_obs(), reward, terminated, False, info


# ==========================================
# Linear learning rate schedule
# ==========================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ==========================================
# Train + test pipeline for one scenario
# ==========================================
def run_scenario(scenario: dict, total_timesteps: int = 300000) -> dict:
    """
    Train a PPO agent for the given scenario, then run one deterministic test episode.
    Returns the wealth and portfolio-weight trajectory.
    """
    env_kwargs = {k: v for k, v in scenario.items() if k != "name"}

    def make_env():
        return DiscreteAssetAllocationEnv(**env_kwargs)

    # --- Training ---
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    policy_kwargs = dict(optimizer_kwargs=dict(weight_decay=1e-4))
    model = PPO(
        "MlpPolicy", venv, verbose=0,
        learning_rate=linear_schedule(0.001),
        n_steps=1024, batch_size=64,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)

    # --- Save / reload with eval settings (temp files, no side effects) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model")
        stats_path = os.path.join(tmpdir, "stats.pkl")
        model.save(model_path)
        venv.save(stats_path)

        eval_venv = DummyVecEnv([make_env])
        eval_venv = VecNormalize.load(stats_path, eval_venv)
        eval_venv.training = False   # freeze running mean/var
        eval_venv.norm_reward = False
        loaded_model = PPO.load(model_path, env=eval_venv)

        # --- Deterministic test episode ---
        obs = eval_venv.reset()
        raw_env = eval_venv.venv.envs[0]  # unwrap to access true W and p

        history_W = [raw_env.W]
        history_p = [raw_env.p.copy()]
        done = False

        print(f"\n{'='*60}")
        print(f"Scenario : {scenario['name']}")
        print(f"Params   : n={env_kwargs['n']}, T={env_kwargs['T']}, "
              f"r={env_kwargs['r']}, lambda={env_kwargs['aversion_rate']}")
        print(f"{'='*60}")

        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            # 注意：这里会返回 infos
            obs, _, dones, infos = eval_venv.step(action) 
            done = dones[0]
            
            if done:
                # 如果结束了，从 info 中提取被自动重置前的数据
                true_W = infos[0]["terminal_W"]
                true_p = infos[0]["terminal_p"]
                
                history_W.append(true_W)
                history_p.append(true_p)
                print(f"  t={env_kwargs['T']:2d} | W={true_W:.4f} | p={np.round(true_p, 3)}")
                print(f"  --> Terminal Wealth W(T): {true_W:.4f}")
            else:
                # 如果没结束，正常读取
                history_W.append(raw_env.W)
                history_p.append(raw_env.p.copy())
                print(f"  t={raw_env.t:2d} | W={raw_env.W:.4f} | p={np.round(raw_env.p, 3)}")

        eval_venv.close()

    venv.close()
    return {
        "name": scenario["name"],
        "T": env_kwargs["T"],
        "n": env_kwargs["n"],
        "history_W": history_W,
        "history_p": np.array(history_p),
    }


# ==========================================
# Main: demonstrate generality across
# different n, T, r, a(k), s(k), p_init
# ==========================================
if __name__ == "__main__":
    scenarios = [
        # Scenario 1: minimum valid n, short horizon, default params
        {
            "name": "n=3, T=5 | default params",
            "n": 3, "T": 5, "r": 0.02, "aversion_rate": 1.0,
        },
        # Scenario 2: maximum valid n, longer horizon, default params
        {
            "name": "n=4, T=8 | default params",
            "n": 4, "T": 8, "r": 0.02, "aversion_rate": 1.0,
        },
        # Scenario 3: near-maximum horizon, higher risk-free rate, higher risk aversion
        {
            "name": "n=3, T=9 | high r, high aversion",
            "n": 3, "T": 9, "r": 0.05, "aversion_rate": 2.0,
        },
        # Scenario 4: fully custom a(k), s(k), p_init — verifies parameter generality
        {
            "name": "n=4, T=6 | custom a, s, p_init",
            "n": 4, "T": 6, "r": 0.01,
            "a": [0.06, 0.09, 0.11, 0.14],
            "s": [0.005, 0.012, 0.022, 0.035],
            "p_init": [0.25, 0.25, 0.20, 0.15, 0.15],
            "aversion_rate": 1.5,
        },
    ]

    results = []
    for sc in scenarios:
        result = run_scenario(sc, total_timesteps=300000)
        results.append(result)

    # ==========================================
    # Plot: wealth + portfolio weights per scenario
    # ==========================================
    n_sc = len(results)
    fig, axes = plt.subplots(n_sc, 2, figsize=(14, 4 * n_sc))

    for i, res in enumerate(results):
        T, n = res["T"], res["n"]
        ax_w, ax_p = axes[i]

        ax_w.plot(range(T + 1), res["history_W"], marker="o", color="steelblue")
        ax_w.set_title(f"[{res['name']}]  Wealth W(t)")
        ax_w.set_ylabel("Wealth")
        ax_w.grid(True)

        labels = ["Cash"] + [f"Asset {k}" for k in range(1, n + 1)]
        for j in range(n + 1):
            ax_p.plot(range(T + 1), res["history_p"][:, j], marker="s", label=labels[j])
        ax_p.set_title(f"[{res['name']}]  Portfolio Weights")
        ax_p.set_xlabel("Time Step")
        ax_p.set_ylabel("Weight")
        ax_p.legend(fontsize=8)
        ax_p.grid(True)

    plt.tight_layout()
    plt.savefig("results.png", dpi=120)
    plt.show()
    print("\nPlot saved to results.png")
