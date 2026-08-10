"""Entrenamiento de SARSA on-policy sobre BinaryMathEnv (CPU).

Uso:
    python train_sarsa.py [--bits 2] [--height 2] [--episodes 1000]
                          [--alpha 0.1] [--gamma 0.95] [--seed 42] [--no-plot]

Artifacts en out/sarsa/:
    returns.csv         recompensas por episodio (SARSA y baseline)
    Q.npy               tabla Q aprendida
    learning_curve.png  curva de aprendizaje vs baseline aleatorio
    best_multiplier.v .. best_multiplier_5.v  Top 5 de circuitos
"""

import argparse
import os

import numpy as np

from envs import BinaryMathEnv
from agents.cpu.sarsa import SarsaAgentCPU
from training.baseline import random_baseline_cpu
from training.artifacts import TopK, save_q, save_returns, save_topk
from training.plot import learning_curve

OUT_DIR = "out/sarsa"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
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
    baseline = random_baseline_cpu(env, 200, base_seed=args.seed + 1_000_000)
    print(f"Baseline aleatorio: mean={baseline.mean():.2f} "
          f"best={baseline.max():.2f}")

    # Entrenamiento SARSA. Sembrar np.random para que closed() use la MISMA
    # secuencia de casos de prueba en cada corrida (reproducibilidad).
    np.random.seed(args.seed)
    agent = SarsaAgentCPU(
        n_states=n_states,
        n_actions=n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        rng_seed=args.seed + 2,
    )

    best_return = -np.inf
    best_ep = -1
    topk = TopK(k=5)

    def on_episode(ep, return_, epsilon):
        nonlocal best_return, best_ep
        if return_ > best_return:
            best_return = return_
            best_ep = ep
        topk.add(return_, tuple(env.suma_grid))
        if ep % 100 == 0 or ep == args.episodes - 1:
            print(f"ep={ep:4d}  eps={epsilon:.3f}  "
                  f"return={return_:8.2f}  best={best_return:8.2f}")

    returns, epsilons = agent.train(
        env, args.episodes, base_seed=args.seed, on_episode=on_episode,
        early_stop_return=100.0,
    )

    print(f"Episodios ejecutados: {len(returns)}")

    # Evaluación greedy con semillas frescas (generalización)
    np.random.seed(args.seed + 3)
    eval_returns, eval_errors = agent.evaluate(env, 50, base_seed=args.seed + 2_000_000)

    print("\n=== Resultados SARSA (CPU) ===")
    print(f"Best training:  return={best_return:.2f} en ep={best_ep}")
    print(f"Eval greedy:    mean={eval_returns.mean():.2f}  "
          f"best={eval_returns.max():.2f}  mean_error={np.nanmean(eval_errors):.4f}")
    print(f"SARSA vs baseline: {eval_returns.mean() - baseline.mean():+.2f}")

    save_q(os.path.join(OUT_DIR, "Q.npy"), agent.Q)
    records = list(zip(range(len(returns)), epsilons, returns))
    save_returns(os.path.join(OUT_DIR, "returns.csv"), records)
    save_topk(topk, OUT_DIR, args.bits, args.height, "best_multiplier",
              grid_from_key=lambda key: list(key))

    if not args.no_plot:
        learning_curve(returns, baseline, eval_returns.mean(),
                       os.path.join(OUT_DIR, "learning_curve.png"),
                       title="SARSA on-policy - BinaryMathEnv",
                       series_label="SARSA",
                       optimum=100.0, optimum_label="Óptimo (reward 100)")


if __name__ == "__main__":
    main()
