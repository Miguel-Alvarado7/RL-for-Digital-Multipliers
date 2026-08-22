"""Curva de aprendizaje genérica (media móvil vs baseline y eval greedy)."""

import numpy as np


def learning_curve(returns, baseline, eval_mean, path, title,
                   ylabel="Retorno (reward)", series_label="Agente",
                   optimum=None, optimum_label="Óptimo", xlabel="Episodio",
                   point_label=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = np.asarray(returns)
    window = max(10, len(returns) // 50)
    # Media móvil por sumas acumuladas: np.convolve es O(n·w) y con cientos de
    # miles de episodios (n_envs alto) eso domina el tiempo del script.
    csum = np.concatenate([[0.0], np.cumsum(returns, dtype=np.float64)])
    smooth = (csum[window:] - csum[:-window]) / window

    # La nube de puntos crudos se submuestrea: dibujar 250k marcadores tarda
    # más que todo el entrenamiento y no aporta resolución visible.
    step = max(1, len(returns) // 5000)
    idx = np.arange(0, len(returns), step)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, returns[idx], alpha=0.15, color="tab:blue",
            label=point_label or f"{series_label} por episodio")
    ax.plot(np.arange(window - 1, len(returns)), smooth, color="tab:blue",
            label=f"{series_label} media móvil (w={window})")
    ax.axhline(baseline.mean(), color="tab:gray", ls="--",
               label=f"Baseline aleatorio {baseline.mean():.2f}")
    ax.axhline(eval_mean, color="tab:green", ls=":",
               label=f"Eval greedy {eval_mean:.2f}")
    if optimum is not None:
        ax.axhline(optimum, color="tab:red", ls="-.", lw=0.8, label=optimum_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Curva de aprendizaje guardada en {path}")
