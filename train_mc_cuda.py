"""Entrenamiento de Monte Carlo on-policy batched sobre BinaryMathEnvCUDAOptimized.

A diferencia de train_mc.py (CPU, un entorno secuencial evaluado con
iverilog/vvp por episodio), este script ejecuta `n_envs` episodios en
paralelo con operaciones vectorizadas de torch:

    python train_mc_cuda.py [--bits 2] [--height 2] [--n-envs 1024]
                            [--episodes 262144] [--alpha 0.1] [--seed 42]
                            [--device cuda] [--no-plot]

Si no hay GPU, torch cae a CPU automáticamente (los tiempos bajan, pero el
código es idéntico). El reward CUDA está en [-10, 0] con 0 = circuito
perfecto, así que el "éxito" es acercarse a 0 (el agente CPU busca 100).

Artifacts en out/:
    returns_cuda.csv        retornos por episodio (MC y baseline)
    Q_cuda.npy              tabla Q aprendida
    learning_curve_cuda.png curva de aprendizaje vs baseline aleatorio
    best_multiplier_cuda.v  Verilog del mejor circuito encontrado
"""

import argparse
import os

import numpy as np
import torch

from Environment.Environment import BinaryMathEnv
from Environment.Environment import BinaryMathEnvCUDAOptimized
from agents.montecarlo_cuda import MonteCarloAgentCUDA

OUT_DIR = "out"


def random_baseline(env, n_batches, seed):
    """Política aleatoria uniforme como referencia (batched).

    Reutiliza el sampler del agente con epsilon=1.0 (exploración pura),
    ejecutando n_batches de n_envs episodios en paralelo.

    Returns:
        (n_batches*n_envs,) float — retornos de cada episodio aleatorio.
    """
    agent = MonteCarloAgentCUDA(
        n_states=env.CC + 1,
        n_actions=env.n_actions,
        device=env.device.type,
        seed=seed,
    )
    returns_all = []
    for _ in range(n_batches):
        _, _, returns = agent.collect_batch(env, epsilon=1.0)
        returns_all.append(returns)
    return torch.cat(returns_all).cpu().numpy()


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
    #   es lo que da la aceleración (toda la evaluación es vectorizada).
    #   Un batch es un único GEMM: medido, n_envs=64 y n_envs=4096 tardan lo
    #   MISMO (la GPU está ociosa por debajo), así que valores pequeños solo
    #   desperdician throughput. El coste en memoria es O(n_envs·chunk_size),
    #   y chunk_size se auto-ajusta a la baja para compensar.
    parser.add_argument(
        "--n-envs", type=int, default=1024,
        help="Entornos paralelos por batch (default: 1024). Valores menores "
             "no aceleran nada: la GPU se satura muy por encima de eso.",
    )

    # --episodes: presupuesto TOTAL de episodios de entrenamiento.
    #   Como cada batch produce n_envs episodios, el número real de batches es
    #   ceil(episodes / n_envs). epsilon decae linealmente por batch.
    parser.add_argument(
        "--episodes", type=int, default=262144,
        help="Presupuesto total de episodios (default: 262144 = 256 batches "
             "de 1024). Se ejecutan ceil(episodes/n_envs) batches; lo que "
             "gobierna el aprendizaje es el NÚMERO DE BATCHES (updates de Q), "
             "así que al subir --n-envs hay que subir esto en proporción.",
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

    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

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
    baseline = random_baseline(env, min(args.baseline_batches, n_batches),
                               seed=args.seed + 1_000_000)
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
    best_grid = None
    best_ep = -1

    def on_batch(b, epsilon, returns_tensor, greedy_info=None,
                 batch_grids=None):
        nonlocal best_return, best_grid, best_ep
        batch_max = returns_tensor.max().item()
        if batch_max > best_return:
            best_return = batch_max
            best_ep = b
            bi = int(returns_tensor.argmax().item())
            # Usar la rejilla del batch de comportamiento; env.suma_grid ya
            # pudo haber sido sobrescrito por la corrida greedy.
            best_grid = env.grid_to_state(batch_grids[bi])
        if b % 100 == 0 or b == n_batches - 1:
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
    early_stop_val = None if args.early_stop.strip().lower() in ("none", "off") \
        else float(args.early_stop)

    records, stopped = agent.train(
        env, n_batches, on_batch=on_batch, early_stop_return=early_stop_val,
        greedy_eval_every=10,
    )

    print(f"Batches ejecutados: {len(records) // env.n_envs}  "
          f"(early_stop_politica_greedy={'si' if stopped else 'no'})")

    # =========================================================================
    # Evaluación greedy con episodios frescos (generalización)
    # =========================================================================
    eval_returns = agent.evaluate(env)
    print("\n=== Resultados MC CUDA ===")
    print(f"Best training:  reward={best_return:.3f} en batch={best_ep}  "
          f"(0 = circuito perfecto)")
    # La política greedy es determinista: un único retorno la caracteriza.
    print(f"Eval greedy:    return={eval_returns.mean():.3f}")
    print(f"MC vs baseline: {eval_returns.mean() - baseline.mean():+.3f}")

    # =========================================================================
    # Artefactos
    # =========================================================================
    np.save(os.path.join(OUT_DIR, "Q_cuda.npy"), agent.Q.cpu().numpy())

    header = "episode,epsilon,mc_return"
    data = np.array(records, dtype=float)
    np.savetxt(os.path.join(OUT_DIR, "returns_cuda.csv"), data,
               delimiter=",", header=header, comments="")

    if best_grid is not None:
        env_best = BinaryMathEnv(Bits=args.bits, Proof=8, height=args.height)
        env_best.suma_grid = list(best_grid["suma_grid"])
        env_best.generate_verilog()
        with open(os.path.join(OUT_DIR, "best_multiplier_cuda.v"), "w") as f:
            f.write(env_best.last_verilog_code)
        print(f"Best Verilog guardado en "
              f"{os.path.join(OUT_DIR, 'best_multiplier_cuda.v')}")

    if not args.no_plot:
        _plot_learning_curve(np.array([r[2] for r in records]), baseline,
                             eval_returns.mean(),
                             os.path.join(OUT_DIR, "learning_curve_cuda.png"))


def _plot_learning_curve(returns, baseline, eval_mean, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Media móvil por sumas acumuladas: np.convolve es O(n·w) y con cientos de
    # miles de episodios (n_envs alto) eso domina el tiempo del script.
    window = max(10, len(returns) // 50)
    csum = np.concatenate([[0.0], np.cumsum(returns, dtype=np.float64)])
    smooth = (csum[window:] - csum[:-window]) / window

    # La nube de puntos crudos se submuestrea: dibujar 250k marcadores tarda
    # más que todo el entrenamiento y no aporta resolución visible.
    step = max(1, len(returns) // 5000)
    idx = np.arange(0, len(returns), step)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, returns[idx], alpha=0.15, color="tab:blue",
            label="MC por episodio")
    ax.plot(np.arange(window - 1, len(returns)), smooth, color="tab:blue",
            label=f"MC media móvil (w={window})")
    ax.axhline(baseline.mean(), color="tab:gray", ls="--",
               label=f"Baseline aleatorio {baseline.mean():.2f}")
    ax.axhline(eval_mean, color="tab:green", ls=":",
               label=f"Eval greedy {eval_mean:.2f}")
    ax.axhline(0.0, color="tab:red", ls="-.", lw=0.8,
               label="Óptimo (reward 0)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Retorno (reward CUDA, [-10, 0])")
    ax.set_title("Monte Carlo on-policy batched - BinaryMathEnvCUDAOptimized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Curva de aprendizaje guardada en {path}")


if __name__ == "__main__":
    main()
