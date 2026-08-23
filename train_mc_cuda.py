"""Entrenamiento de Monte Carlo on-policy batched sobre BinaryMathEnvCUDAOptimized.

A diferencia de train_mc.py (CPU, un entorno secuencial evaluado con
iverilog/vvp por episodio), este script ejecuta `n_envs` episodios en
paralelo con operaciones vectorizadas de torch:

    python train_mc_cuda.py [--bits 2] [--height 2] [--n-envs 512]
                            [--episodes 100000] [--alpha 0.1] [--seed 42]
                            [--device cuda] [--no-plot]

Si no hay GPU, torch cae a CPU automáticamente (los tiempos bajan, pero el
código es idéntico). El reward CUDA está en [-10, 0] con 0 = circuito
perfecto, así que el "éxito" es acercarse a 0 (el agente CPU busca 100).

El entorno optimized evalúa cada rejilla con un único GEMM (action_vals
cacheado), procesando los casos de prueba por chunks para acotar la memoria.
Un batch es un solo GEMM: en GPU n_envs=64 y n_envs=4096 tardan lo MISMO (la
GPU está ociosa por debajo), así que valores pequeños solo desperdician
throughput. Si hay OOM, chunk_size se auto-ajusta a la baja.

Artifacts en out/montecarlo_cuda/bits<N>/ (N = --bits, un directorio por ancho):
    returns_cuda.csv        retornos por episodio (MC y baseline)
    Q_cuda.npy              tabla Q aprendida
    learning_curve_cuda.png curva de aprendizaje vs baseline aleatorio
    best_multiplier_cuda.v  Verilog del mejor circuito (rank 1 del top 5)
    best_multiplier_cuda_2.v.._5.v  Verilog de los circuitos 2º a 5º
"""

import argparse
import os

import numpy as np
import torch

from envs import BinaryMathEnv
from envs import BinaryMathEnvCUDAOptimized
from agents.cuda.montecarlo import MonteCarloAgentCUDA
from training.artifacts import (TopK, clear_checkpoints, save_q, save_returns,
                                save_topk, save_topk_state, topk_add_batch)
from training.baseline import random_baseline_cuda
from training.plot import learning_curve

OUT_DIR = "out/montecarlo_cuda/bits{N}"


def main():
    # =========================================================================
    # Parser de argumentos
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Monte Carlo on-policy batched sobre BinaryMathEnvCUDAOptimized."
    )

    # --bits: ancho del multiplicador (bits de A y de B).
    #   Define:
    #     - el espacio de acciones: 2 + 4·bits² (0, 1 y productos parciales
    #       A[i]&B[j] con sus 4 combinaciones de negación ~);
    #     - el número de casos de prueba EXHAUSTIVOS: (2^bits)². Para bits=2
    #       son 16 pares; para bits=8, 65536 (de ahí el streaming por chunks
    #       del entorno optimized).
    parser.add_argument(
        "--bits", type=int, default=2,
        help="Bits del multiplicador A y B (default: 2). 2 => 16 casos, "
             "4 => 256 casos, 8 => 65536 casos exhaustivos.",
    )

    # --height: filas de la tabla de productos parciales.
    #   La tabla tiene height × (2·bits) celdas; cada episodio rellena
    #   CC = height·2·bits celdas. height mayor => tabla más grande => episodio
    #   más largo y más grados de libertad para el circuito.
    parser.add_argument(
        "--height", type=int, default=2,
        help="Filas de la tabla de productos parciales (default: 2). "
             "Celdas por episodio = height·2·bits.",
    )

    # --n-envs: entornos en paralelo por batch.
    #   Cada batch equivale a n_envs episodios MC independientes. En GPU esto
    #   es lo que da la aceleración (toda la evaluación es vectorizada). Un
    #   batch es un único GEMM: n_envs=64 y n_envs=4096 tardan lo mismo, así
    #   que valores pequeños solo desperdician throughput. El coste en memoria
    #   es O(n_envs·chunk_size) y chunk_size se auto-ajusta a la baja.
    parser.add_argument(
        "--n-envs", type=int, default=512,
        help="Entornos paralelos por batch (default: 512). Reduce si hay OOM; "
             "subir no acelera por debajo de ~1024 en GPU.",
    )

    # --episodes: presupuesto TOTAL de episodios de entrenamiento.
    #   Como cada batch produce n_envs episodios, el número real de batches es
    #   ceil(episodes / n_envs). epsilon decae linealmente por batch.
    parser.add_argument(
        "--episodes", type=int, default=100000,
        help="Presupuesto total de episodios (default: 100000). Se ejecutan "
             "ceil(episodes/n_envs) batches; al subir --n-envs hay que subir "
             "esto en proporción (lo que gobierna es el Nº de batches).",
    )

    # --baseline-batches: batches de política aleatoria para la referencia.
    #   Antes se corrían tantos batches como el entrenamiento (~50% del
    #   cómputo total del script) para dibujar una línea horizontal.
    parser.add_argument(
        "--baseline-batches", type=int, default=8,
        help="Batches de baseline aleatorio (default: 8). Se limita a "
             "n_batches si este es menor.",
    )

    # --alpha: tasa de aprendizaje de la actualización MC:
    #   Q[s,a] += alpha·(G − Q[s,a]). alpha alto => converge rápido pero
    #   oscila; alpha bajo => estable pero lento.
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Tasa de aprendizaje MC (default: 0.1).",
    )

    # --seed: semilla global de numpy y torch. Controla el muestreo de
    #   acciones (epsilon-greedy y baseline). Los casos de prueba son
    #   exhaustivos y fijos, así que no aportan aleatoriedad.
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla de numpy/torch para reproducibilidad (default: 42).",
    )

    # --device: dónde correr los tensores. 'cuda' usa la GPU si está
    #   disponible; 'cpu' fuerza CPU. El agente y el entorno caen a CPU
    #   automáticamente si se pide cuda sin GPU.
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Dispositivo de cómputo (default: cuda, con fallback a CPU).",
    )

    # --no-plot: omite la generación de la curva de aprendizaje (útil en
    #   servidores sin backend gráfico o para ejecuciones rápidas).
    parser.add_argument(
        "--no-plot", action="store_true",
        help="No generar learning_curve_cuda.png.",
    )

    # --early-stop: umbral de retorno para detener el entrenamiento cuando
    #   algún episodio del batch lo alcanza o supera (retorno CUDA en [-10, 0],
    #   0 = circuito perfecto). Acepta:
    #     - un número (p. ej. -0.5): detiene al alcanzar ese umbral;
    #     - 'none' / 'off': desactiva el early stop (None en el agente), forzando
    #       a ejecutar siempre los n_batches completos.
    #   Valor por defecto: 0.0 (detiene en cuanto hay un circuito perfecto).
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

    out_dir = OUT_DIR.format(N=args.bits)
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # Entorno CUDA batched
    # =========================================================================
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

    n_batches = int(np.ceil(args.episodes / args.n_envs))

    # Baseline aleatorio (misma escala de reward CUDA: [-10, 0]).
    np.random.seed(args.seed + 1)
    baseline = random_baseline_cuda(
        MonteCarloAgentCUDA, env,
        min(args.baseline_batches, n_batches), seed=args.seed + 1_000_000,
    )
    print(f"Baseline aleatorio: mean={baseline.mean():.2f} "
          f"best={baseline.max():.2f}")

    # =========================================================================
    # Entrenamiento MC batched
    # =========================================================================
    torch.manual_seed(args.seed)
    agent = MonteCarloAgentCUDA(
        n_states=n_states,
        n_actions=n_actions,
        alpha=args.alpha,
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
        # Actualizar Top-5 con las rejillas del batch de comportamiento
        # (snapshot antes de la greedy; env.suma_grid pudo haber sido
        # sobrescrito por la corrida greedy).
        if batch_grids is not None:
            topk_add_batch(topk, returns_tensor, batch_grids)
        if args.log_every > 0 and (b % args.log_every == 0
                                   or b == n_batches - 1):
            # La media solo se materializa cuando se va a imprimir: cada
            # .item() sincroniza CPU<->GPU.
            print(f"batch={b:4d}/{n_batches}  eps={epsilon:.3f}  "
                  f"mean={returns_tensor.mean().item():8.3f}  "
                  f"best={best_return:8.3f}")
        # Monitoreo de la política greedy (corrida ε=0, determinista).
        if greedy_info is not None:
            g_mean = greedy_info['mean']
            state = "confirmado" if greedy_info['confirmed'] else "no confirmado"
            print(f"    greedy: batch={b:4d}  mean={g_mean:8.3f}  "
                  f"candidato={'si' if greedy_info['candidate'] else 'no'}  "
                  f"estado={state}")

    # Convertir --early-stop: 'none'/'off' => None (desactivado), si no => float.
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

    # =========================================================================
    # Evaluación greedy (la política es determinista: un retorno la caracteriza)
    # =========================================================================
    eval_returns = agent.evaluate(env)
    print("\n=== Resultados MC CUDA ===")
    print(f"Best training:  reward={best_return:.3f} en batch={best_ep}  "
          f"(0 = circuito perfecto)")
    print(f"Eval greedy:    return={eval_returns.mean():.3f}")
    print(f"MC vs baseline: {eval_returns.mean() - baseline.mean():+.3f}")

    # =========================================================================
    # Artefactos
    # =========================================================================
    save_q(os.path.join(out_dir, "Q_cuda.npy"), agent.Q)
    save_returns(os.path.join(out_dir, "returns_cuda.csv"), stats)

    save_topk(
        topk, out_dir, args.bits, args.height, "best_multiplier_cuda",
        grid_from_key=lambda key: env.grid_to_state(list(key))['suma_grid'],
    )

    # Los artefactos definitivos ya estan en disco, asi que los checkpoints
    # sobran. Se borra DESPUES de guardarlos: si el proceso muriera durante el
    # volcado final, el checkpoint seguiria siendo la ultima copia buena.
    clear_checkpoints(out_dir)

    if not args.no_plot:
        learning_curve(
            stats[:, 3], baseline, eval_returns.mean(),
            os.path.join(out_dir, "learning_curve_cuda.png"),
            title="Monte Carlo on-policy batched - BinaryMathEnvCUDAOptimized",
            ylabel="Retorno (reward CUDA, [-10, 0])",
            series_label="MC",
            xlabel="Batch", point_label="MC media por batch", optimum=0.0, optimum_label="Óptimo (reward 0)",
            best_series=stats[:, 6], max_series=stats[:, 6],
        )


if __name__ == "__main__":
    main()
