"""Entrenamiento de Q-learning off-policy batched sobre BinaryMathEnvCUDAOptimized.

Uso:
    python train_qlearning_cuda.py [--bits 2] [--height 2] [--n-envs 512]
                                   [--episodes 100000] [--alpha 0.1] [--gamma 0.95]
                                   [--seed 42] [--device cuda] [--no-plot]

Artifacts en out/qlearning_cuda/bits{N}/ (N = número de bits):
    returns_cuda.csv            retornos por episodio
    Q_cuda.npy                  tabla Q aprendida
    learning_curve_cuda.png     curva de aprendizaje vs baseline aleatorio
    best_multiplier_cuda.v .. best_multiplier_cuda_5.v  Top 5 de circuitos
"""

import argparse
import math
import os

import numpy as np
import torch

from envs import BinaryMathEnvCUDAOptimized
from agents.cuda.qlearning import QLearningAgentCUDA
from training.baseline import random_baseline_cuda
from training.artifacts import TopK, save_q, save_returns, save_topk
from training.plot import learning_curve

OUT_DIR = "out/qlearning_cuda/bits{N}"


def main():
    parser = argparse.ArgumentParser(
        description="Q-learning off-policy batched sobre BinaryMathEnvCUDAOptimized."
    )
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--baseline-batches", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--early-stop", type=str, default="0.0",
        help="Umbral de early stop (default: 0.0). 'none'/'off' lo desactiva.",
    )
    args = parser.parse_args()

    out_dir = OUT_DIR.format(N=args.bits)
    os.makedirs(out_dir, exist_ok=True)

    env = BinaryMathEnvCUDAOptimized(
        Bits=args.bits,
        height=args.height,
        n_envs=args.n_envs,
        device=args.device,
    )
    n_actions = env.n_actions
    n_states = env.CC + 1

    print(f"Config: Bits={args.bits} height={args.height} "
          f"celdas={env.CC} acciones={n_actions} "
          f"device={env.device} n_envs={env.n_envs}")

    n_batches = math.ceil(args.episodes / args.n_envs)

    # Baseline aleatorio (misma escala de reward CUDA: [-10, 0]).
    np.random.seed(args.seed + 1)
    baseline = random_baseline_cuda(QLearningAgentCUDA, env,
                                    min(args.baseline_batches, n_batches),
                                    seed=args.seed + 1_000_000)
    print(f"Baseline aleatorio: mean={baseline.mean():.2f} "
          f"best={baseline.max():.2f}")

    torch.manual_seed(args.seed)
    agent = QLearningAgentCUDA(
        n_states=n_states,
        n_actions=n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        device=args.device,
        seed=args.seed + 2,
    )

    best_return = -np.inf
    best_ep = -1
    topk = TopK(k=5)

    def on_batch(b, epsilon, returns_tensor, greedy_info=None,
                 batch_grids=None):
        nonlocal best_return, best_ep, topk
        batch_max = returns_tensor.max().item()
        if batch_max > best_return:
            best_return = batch_max
            best_ep = b
        # Top-5 con las rejillas del batch de comportamiento (snapshot antes
        # de la greedy; env.suma_grid pudo haber sido sobrescrito).
        if batch_grids is not None:
            ret_cpu = returns_tensor.cpu()
            for i in range(env.n_envs):
                topk.add(ret_cpu[i].item(), tuple(batch_grids[i].tolist()))
        if b % 100 == 0 or b == n_batches - 1:
            print(f"batch={b:4d}/{n_batches}  eps={epsilon:.3f}  "
                  f"mean={returns_tensor.mean().item():8.3f}  "
                  f"best={best_return:8.3f}")
        if greedy_info is not None:
            g_mean = greedy_info['mean']
            state = "confirmado" if greedy_info['confirmed'] else "no confirmado"
            print(f"    greedy: batch={b:4d}  mean={g_mean:8.3f}  "
                  f"candidato={'si' if greedy_info['candidate'] else 'no'}  "
                  f"estado={state}")

    early_stop_val = None if args.early_stop.strip().lower() in ("none", "off") \
        else float(args.early_stop)

    records, stopped = agent.train(
        env, n_batches, on_batch=on_batch, early_stop_return=early_stop_val,
        greedy_eval_every=10,
    )

    print(f"Batches ejecutados: {len(records) // env.n_envs}  "
          f"(early_stop_politica_greedy={'si' if stopped else 'no'})")

    eval_returns = agent.evaluate(env)
    print("\n=== Resultados Q-learning CUDA ===")
    print(f"Best training:  reward={best_return:.3f} en batch={best_ep}  "
          f"(0 = circuito perfecto)")
    print(f"Eval greedy:    return={eval_returns.mean():.3f}")
    print(f"Q-learning vs baseline: {eval_returns.mean() - baseline.mean():+.3f}")

    save_q(os.path.join(out_dir, "Q_cuda.npy"), agent.Q)
    save_returns(os.path.join(out_dir, "returns_cuda.csv"), records)
    save_topk(topk, out_dir, args.bits, args.height, "best_multiplier_cuda",
              grid_from_key=lambda key: env.grid_to_state(list(key))["suma_grid"])

    if not args.no_plot:
        learning_curve([r[2] for r in records], baseline, eval_returns.mean(),
                       os.path.join(out_dir, "learning_curve_cuda.png"),
                       title="Q-learning off-policy batched - BinaryMathEnvCUDAOptimized",
                       ylabel="Retorno (reward CUDA, [-10, 0])",
                       series_label="Q-learning",
                       optimum=0.0, optimum_label="Óptimo (reward 0)")


if __name__ == "__main__":
    main()
