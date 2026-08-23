"""Regenera las curvas de aprendizaje desde los artefactos de out/, sin reentrenar.

Para cada corrida con un returns_cuda.csv en formato por batch reconstruye
el grafico completo:

  - curva azul: columna `mean` del CSV (nube submuestreada + media movil),
    identica a la que escribieron los scripts de entrenamiento;
  - linea naranja: media movil de la columna `max` ("mejor episodio por
    batch"), suavizada con la misma ventana que la media azul — visualiza
    cuando la EXPLORACION alcanzo el optimo aunque la media del
    comportamiento no lo muestre;
  - linea verde "Eval greedy": recuperada EXACTA desde Q_cuda.npy (argmax de
    Q por cursor, evaluado en el entorno en CPU), sin depender del log;
  - linea gris "Baseline aleatorio": solo con --with-baseline, REGENERADA con
    la semilla del entrenamiento (--seed, 42 por defecto). OJO: solo es fiel
    bit a bit si se replica el device de la corrida original, porque el RNG
    de torch difiere entre CPU y CUDA; en otro caso es una aproximacion.

Los originales nunca se tocan: la salida se llama learning_curve_cuda_v2.png
junto al CSV; --overwrite escribe sobre learning_curve_cuda.png.

Uso:
    python plot_from_artifacts.py                     # todas las corridas
    python plot_from_artifacts.py --runs out/qlearning_cuda/bits2
    python plot_from_artifacts.py --with-baseline --overwrite

Los CSV de formato antiguo (una fila por episodio) y los directorios que no
casan con bits<N>[_nsteps<M>] se saltan con aviso.
"""

import argparse
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import torch

from envs import BinaryMathEnvCUDAOptimized
from training.baseline import random_baseline_cuda
from agents.cuda.montecarlo import MonteCarloAgentCUDA

#: Cabecera esperada del formato agregado por batch. Cualquier otra (p. ej.
#: episode,epsilon,mc_return del formato antiguo) hace saltar la corrida.
BATCH_HEADER = "batch,episode_start,epsilon,mean,std,min,max,p25,p50,p75"

SERIES_BY_PARENT = {
    "montecarlo_cuda": "Monte Carlo",
    "qlearning_cuda": "Q-learning",
    "sarsa_cuda": "SARSA",
}


def _moving_average(x):
    """(ventana, media movil) con la misma receta que training.plot."""
    x = np.asarray(x, dtype=np.float64)
    window = max(10, len(x) // 50)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    return window, (csum[window:] - csum[:-window]) / window


def learning_curve_smooth_best(returns, baseline, eval_mean, path, title,
                               ylabel, series_label, xlabel, point_label,
                               optimum, optimum_label, best_series=None,
                               max_series=None):
    """Igual que training.plot.learning_curve pero la linea naranja es la
    media movil del mejor episodio por batch (sin maximo acumulado). Si se
    provee max_series, se dibuja una línea en color púrpura con los valores
    máximos por batch (igual que la media azul pero para el máximo)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window, smooth_mean = _moving_average(returns)
    _, smooth_best = _moving_average(best_series)

    step = max(1, len(returns) // 5000)
    idx = np.arange(0, len(returns), step)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, np.asarray(returns)[idx], alpha=0.15, color="tab:blue",
            label=point_label)
    ax.plot(np.arange(window - 1, len(returns)), smooth_mean, color="tab:blue",
            label=f"{series_label} media móvil (w={window})")
    ax.plot(np.arange(window - 1, len(best_series)), smooth_best,
            color="tab:orange", lw=2, alpha=0.7,
            label=f"Mejor episodio media móvil (w={window})")
    if max_series is not None:
        ax.plot(idx, np.asarray(max_series)[idx], color="tab:orange", alpha=0.2,
                lw=1.5,
                label="Mejor episodio por batch")
    if baseline is not None:
        ax.axhline(baseline.mean(), color="tab:gray", ls="--",
                   label=f"Baseline aleatorio {baseline.mean():.2f}")
    ax.axhline(eval_mean, color="tab:green", ls=":",
               label=f"Eval greedy {eval_mean:.2f}")
    ax.axhline(optimum, color="tab:red", ls="-.", lw=0.8, label=optimum_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Curva de aprendizaje guardada en {path}")


def parse_run_dir(run: Path):
    """(bits, height_auto, n_steps, etiqueta_serie) o None si no casa.

    La altura automatica sigue al Makefile: height = bits salvo SARSA a 4
    bits, donde SARSA_HEIGHT_4 = 3.
    """
    m = re.fullmatch(r"bits(\d+)(?:_nsteps(\d+))?", run.name)
    if not m:
        return None
    bits = int(m.group(1))
    n_steps = int(m.group(2)) if m.group(2) else None
    height = 3 if ("sarsa" in run.parent.name and bits == 4) else bits
    base = SERIES_BY_PARENT.get(run.parent.name, run.parent.name)
    label = f"{base} (n_steps={n_steps})" if n_steps else base
    return bits, height, n_steps, label


def greedy_return(q_path: Path, bits: int, height: int) -> float:
    """Retorno exacto de la politica greedy almacenada en Q_cuda.npy.

    Replica greedy_return() de TabularAgentCUDA pero leyendo la tabla desde
    disco y evaluando en CPU: el GEMM corre con TF32 desactivado y los
    productos son enteros pequenos, asi que el resultado coincide con el que
    imprimio el script de entrenamiento.
    """
    env = BinaryMathEnvCUDAOptimized(Bits=bits, height=height, n_envs=1,
                                     device="cpu")
    q = np.load(q_path)
    if q.shape[0] < env.CC:
        raise ValueError(f"{q_path}: esperaba >= {env.CC} filas, tiene {q.shape[0]}")
    grid = torch.as_tensor(q[:env.CC].argmax(axis=1), dtype=torch.int64)
    return float(env.evaluate_grids(grid.unsqueeze(0))[0].item())


def regenerate_baseline(args, bits, height, n_envs, n_batches):
    """Baseline aleatorio con las mismas semillas que los scripts de entrenamiento."""
    env = BinaryMathEnvCUDAOptimized(Bits=bits, height=height,
                                     n_envs=max(n_envs, 1), device=args.device)
    np.random.seed(args.seed + 1)
    return random_baseline_cuda(
        MonteCarloAgentCUDA, env, min(args.baseline_batches, n_batches),
        seed=args.seed + 1_000_000)


def main():
    ap = argparse.ArgumentParser(
        description="Regenera learning curves desde out/ sin reentrenar.")
    ap.add_argument("--out-root", default="out",
                    help="Raiz de artefactos (default: out).")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Directorios concretos; default: descubrir todos.")
    ap.add_argument("--height", type=int, default=None,
                    help="Forzar altura de la tabla (default: regla del Makefile).")
    ap.add_argument("--with-baseline", action="store_true",
                    help="Regenerar el baseline aleatorio (linea gris). Sin "
                         "ella, el grafico sale sin esa referencia.")
    ap.add_argument("--baseline-batches", type=int, default=8,
                    help="Batches del baseline regenerado (default: 8).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla asumida del entrenamiento original "
                         "(default: 42, la del Makefile). Solo afecta al baseline.")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="Device para el baseline regenerado "
                         "(default: cuda si hay GPU, si no cpu).")
    ap.add_argument("--suffix", default="_v2",
                    help="Sufijo del PNG de salida (default: _v2).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Escribir sobre learning_curve_cuda.png original.")
    args = ap.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.out_root)
    if args.runs:
        runs = [Path(p) for p in args.runs]
    else:
        runs = sorted({p.parent for p in root.rglob("returns_cuda.csv")})

    done = skipped = 0
    for run in runs:
        info = parse_run_dir(run)
        csv_path = run / "returns_cuda.csv"
        q_path = run / "Q_cuda.npy"
        if info is None or not csv_path.exists():
            print(f"[skip] {run}: no casa con bits<N>[_nsteps<M>] o sin CSV")
            skipped += 1
            continue
        with open(csv_path) as f:
            header = f.readline().strip()
        if header != BATCH_HEADER:
            print(f"[skip] {run}: formato antiguo (sin agregacion por batch)")
            skipped += 1
            continue

        bits, height_auto, _, label = info
        height = args.height if args.height is not None else height_auto
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        means, maxes = data[:, 3], data[:, 6]

        eval_ret = greedy_return(q_path, bits, height)

        baseline = None
        if args.with_baseline:
            # n_envs inferido del salto de episode_start entre batches.
            step = int(round(data[1, 1] - data[0, 1])) if len(data) > 1 else 512
            with contextlib.redirect_stdout(io.StringIO()):
                baseline = regenerate_baseline(args, bits, height, step,
                                               len(data))

        suffix = "" if args.overwrite else args.suffix
        out_png = run / f"learning_curve_cuda{suffix}.png"
        learning_curve_smooth_best(
            means, baseline, eval_ret, str(out_png),
            title=f"{label} batched - BinaryMathEnvCUDAOptimized"
                  f"{'' if args.overwrite else ' (desde artefactos)'}",
            ylabel="Retorno (reward CUDA, [-10, 0])",
            series_label=label,
            xlabel="Batch",
            point_label=f"{label} media por batch",
            optimum=0.0,
            optimum_label="Óptimo (reward 0)",
            best_series=maxes,
            max_series=maxes,
        )
        print(f"[ok]   {out_png}  eval_greedy={eval_ret:.6f}")
        done += 1

    print(f"\n{done} graficos regenerados, {skipped} corridas saltadas.")


if __name__ == "__main__":
    main()
