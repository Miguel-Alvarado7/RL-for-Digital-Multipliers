"""Entrenamiento de SARSA on-policy batched sobre BinaryMathEnvCUDAOptimized.

Uso:
    python train_sarsa_cuda.py [--bits 2] [--height 2] [--n-envs 512]
                               [--episodes 100000] [--alpha 0.1] [--gamma 0.95]
                               [--n-steps 1] [--seed 42] [--device cuda] [--no-plot]

Artifacts en out/sarsa_cuda/bits{N}_nsteps{S}/ (N = bits, S = n_steps):
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
from agents.cuda.sarsa import SarsaAgentCUDA
from training.baseline import random_baseline_cuda
from training.artifacts import (TopK, clear_checkpoints, save_q, save_returns,
                                save_topk, save_topk_state, topk_add_batch)
from training.plot import learning_curve

OUT_DIR = "out/sarsa_cuda/bits{N}_nsteps{S}"


def main():
    parser = argparse.ArgumentParser(
        description="SARSA on-policy batched sobre BinaryMathEnvCUDAOptimized."
    )
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--baseline-batches", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-steps", type=int, default=1,
                        help="Número de pasos de lookahead para el retorno "
                             "n-step SARSA (default: 1 = SARSA(0)).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--early-stop", type=str, default="0.0",
        help="Umbral de early stop (default: 0.0). 'none'/'off' lo desactiva.",
    )
    # --checkpoint-every: cada cuantos batches se vuelca estado parcial.
    #   Una corrida larga que revienta al 95% no deja nada; con esto deja una
    #   tabla Q, la curva hasta ese punto y los mejores circuitos.
    parser.add_argument(
        "--checkpoint-every", type=int, default=5000,
        help="Volcar estado parcial cada N batches (default: 5000, 0 desactiva).",
    )

    # --quiet: reduce la salida por terminal. El monitoreo greedy periodico
    #   imprime una linea cada `greedy_eval_every` batches: con 97k batches son
    #   ~9.800 lineas por corrida, diez veces mas que las de progreso. Su unico
    #   efecto es ese printout (greedy_info no se usa para nada mas), asi que
    #   con --quiet se salta tambien el computo, ahorrando una sincronizacion
    #   CPU<->GPU por evaluacion.
    #
    #   El early stop NO se ve afectado: cuando un episodio alcanza el umbral
    #   sigue disparandose la corrida greedy de confirmacion, y esa linea si se
    #   imprime, porque es rara y significativa.
    parser.add_argument(
        "--quiet", action="store_true",
        help="Omite el monitoreo greedy periodico (y su computo). No afecta "
             "al early stop ni a los artefactos.",
    )

    # --log-every: cada cuantos batches se imprime la linea de progreso.
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="Imprimir progreso cada N batches (default: 100, 0 lo desactiva).",
    )

    args = parser.parse_args()

    out_dir = OUT_DIR.format(N=args.bits, S=args.n_steps)
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
    baseline = random_baseline_cuda(SarsaAgentCUDA, env,
                                    min(args.baseline_batches, n_batches),
                                    seed=args.seed + 1_000_000)
    print(f"Baseline aleatorio: mean={baseline.mean():.2f} "
          f"best={baseline.max():.2f}")

    torch.manual_seed(args.seed)
    agent = SarsaAgentCUDA(
        n_states=n_states,
        n_actions=n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        device=args.device,
        seed=args.seed + 2,
        n_steps=args.n_steps,
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
            topk_add_batch(topk, returns_tensor, batch_grids)
        if args.log_every > 0 and (b % args.log_every == 0
                                   or b == n_batches - 1):
            print(f"batch={b:4d}/{n_batches}  eps={epsilon:.3f}  "
                  f"mean={returns_tensor.mean().item():8.3f}  "
                  f"best={best_return:8.3f}")
        if greedy_info is not None:
            g_mean = greedy_info['mean']
            state = "confirmado" if greedy_info['confirmed'] else "no confirmado"
            print(f"    greedy: batch={b:4d}  mean={g_mean:8.3f}  "
                  f"candidato={'si' if greedy_info['candidate'] else 'no'}  "
                  f"estado={state}")

    def on_checkpoint(n_done, stats_so_far):
        """Vuelca estado parcial. Escritura atomica: ver _atomic_write."""
        save_q(os.path.join(out_dir, "Q_cuda_partial.npy"), agent.Q)
        save_returns(os.path.join(out_dir, "returns_cuda_partial.csv"), stats_so_far)
        save_topk_state(topk, os.path.join(out_dir, "topk_partial.json"))
        print(f"    [checkpoint] batch={n_done}  -> {out_dir}")

    early_stop_val = None if args.early_stop.strip().lower() in ("none", "off") \
        else float(args.early_stop)

    stats, stopped = agent.train(
        env, n_batches, on_batch=on_batch, early_stop_return=early_stop_val,
        greedy_eval_every=0 if args.quiet else 10,
        checkpoint_every=args.checkpoint_every, on_checkpoint=on_checkpoint,
    )

    print(f"Batches ejecutados: {len(stats)}  "
          f"(early_stop_politica_greedy={'si' if stopped else 'no'})")

    eval_returns = agent.evaluate(env)
    print("\n=== Resultados SARSA CUDA ===")
    print(f"Best training:  reward={best_return:.3f} en batch={best_ep}  "
          f"(0 = circuito perfecto)")
    print(f"Eval greedy:    return={eval_returns.mean():.3f}")
    print(f"SARSA vs baseline: {eval_returns.mean() - baseline.mean():+.3f}")

    save_q(os.path.join(out_dir, "Q_cuda.npy"), agent.Q)
    save_returns(os.path.join(out_dir, "returns_cuda.csv"), stats)
    save_topk(topk, out_dir, args.bits, args.height, "best_multiplier_cuda",
              grid_from_key=lambda key: env.grid_to_state(list(key))["suma_grid"])

    # Los artefactos definitivos ya estan en disco, asi que los checkpoints
    # sobran. Se borra DESPUES de guardarlos: si el proceso muriera durante el
    # volcado final, el checkpoint seguiria siendo la ultima copia buena.
    clear_checkpoints(out_dir)

    if not args.no_plot:
        learning_curve(stats[:, 3], baseline, eval_returns.mean(),
                       os.path.join(out_dir, "learning_curve_cuda.png"),
                       title="SARSA on-policy batched - BinaryMathEnvCUDAOptimized",
                       ylabel="Retorno (reward CUDA, [-10, 0])",
                       series_label="SARSA",
                       xlabel="Batch", point_label="SARSA media por batch",
                       optimum=0.0, optimum_label="Óptimo (reward 0)")


if __name__ == "__main__":
    main()
