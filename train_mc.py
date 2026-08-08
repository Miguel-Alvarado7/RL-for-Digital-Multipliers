"""Entrenamiento de Monte Carlo on-policy sobre BinaryMathEnv (CPU).

Uso:
    python train_mc.py [--bits 2] [--height 2] [--episodes 1000]
                       [--alpha 0.1] [--seed 42] [--no-plot]

Artifacts en out/:
    returns.csv         recompensas por episodio (MC y baseline)
    Q.npy               tabla Q aprendida
    learning_curve.png  curva de aprendizaje vs baseline aleatorio
    best_multiplier.v   Verilog del mejor circuito encontrado
"""

import argparse
import os

import numpy as np

from Environment.Environment import BinaryMathEnv
from agents.montecarlo import MonteCarloAgent

OUT_DIR = "out"


def random_baseline(env, n_episodes, base_seed):
    """Política aleatoria uniforme como referencia."""
    returns = np.empty(n_episodes)
    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        terminated = truncated = False
        total = 0.0
        while not (terminated or truncated):
            action = np.random.randint(env.action_space.n)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
        returns[i] = total
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    env = BinaryMathEnv(Bits=args.bits, Proof=8, height=args.height)
    n_actions = env.action_space.n
    n_states = env.CC + 1

    print(f"Config: Bits={args.bits} height={args.height} "
          f"celdas={env.CC} acciones={n_actions}")

    # Baseline aleatorio (secuencia propia de casos de prueba)
    np.random.seed(args.seed + 1)
    baseline = random_baseline(env, 200, base_seed=args.seed + 1_000_000)
    print(f"Baseline aleatorio: mean={baseline.mean():.2f} "
          f"best={baseline.max():.2f}")

    # Entrenamiento MC. Sembrar np.random para que closed() use la MISMA
    # secuencia de casos de prueba en cada corrida (reproducibilidad).
    np.random.seed(args.seed)
    agent = MonteCarloAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=args.alpha,
        rng_seed=args.seed + 2,
    )

    best_return = -np.inf
    best_grid = None
    best_ep = -1

    def on_episode(ep, return_, epsilon):
        nonlocal best_return, best_grid, best_ep
        if return_ > best_return:
            best_return = return_
            best_grid = np.array(env.suma_grid)
            best_ep = ep
        if ep % 100 == 0 or ep == args.episodes - 1:
            print(f"ep={ep:4d}  eps={epsilon:.3f}  "
                  f"return={return_:8.2f}  best={best_return:8.2f}")

    returns, epsilons = agent.train(
        env, args.episodes, base_seed=args.seed, on_episode=on_episode
    )

    # Evaluación greedy con semillas frescas (generalización)
    np.random.seed(args.seed + 3)
    eval_returns, eval_errors = agent.evaluate(env, 50, base_seed=args.seed + 2_000_000)

    print("\n=== Resultados MC ===")
    print(f"Best training:  return={best_return:.2f} en ep={best_ep}")
    print(f"Eval greedy:    mean={eval_returns.mean():.2f}  "
          f"best={eval_returns.max():.2f}  mean_error={np.nanmean(eval_errors):.4f}")
    print(f"MC vs baseline: {eval_returns.mean() - baseline.mean():+.2f}")

    # Guardar artefactos
    np.save(os.path.join(OUT_DIR, "Q.npy"), agent.Q)

    header = "episode,epsilon,mc_return"
    data = np.column_stack([np.arange(len(returns)), epsilons, returns])
    np.savetxt(os.path.join(OUT_DIR, "returns.csv"), data,
               delimiter=",", header=header, comments="")

    if best_grid is not None:
        env_best = BinaryMathEnv(Bits=args.bits, Proof=8, height=args.height)
        env_best.suma_grid = list(best_grid)
        env_best.generate_verilog()
        with open(os.path.join(OUT_DIR, "best_multiplier.v"), "w") as f:
            f.write(env_best.last_verilog_code)
        print(f"Best Verilog guardado en {os.path.join(OUT_DIR, 'best_multiplier.v')}")

    if not args.no_plot:
        _plot_learning_curve(returns, baseline, eval_returns.mean(),
                             os.path.join(OUT_DIR, "learning_curve.png"))


def _plot_learning_curve(returns, baseline, eval_mean, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = max(10, len(returns) // 50)
    kernel = np.ones(window) / window
    smooth = np.convolve(returns, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(returns, alpha=0.15, color="tab:blue", label="MC por episodio")
    ax.plot(np.arange(window - 1, len(returns)), smooth, color="tab:blue",
            label=f"MC media móvil (w={window})")
    ax.axhline(baseline.mean(), color="tab:gray", ls="--",
               label=f"Baseline aleatorio {baseline.mean():.1f}")
    ax.axhline(eval_mean, color="tab:green", ls=":",
               label=f"Eval greedy {eval_mean:.1f}")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Retorno (reward)")
    ax.set_title("Monte Carlo on-policy - BinaryMathEnv")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Curva de aprendizaje guardada en {path}")


if __name__ == "__main__":
    main()
