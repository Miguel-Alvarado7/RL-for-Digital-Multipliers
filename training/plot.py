"""Curva de aprendizaje genérica (media móvil vs baseline y eval greedy)."""

import numpy as np


def learning_curve(returns, baseline, eval_mean, path, title,
                   ylabel="Retorno (reward)", series_label="Agente",
                   optimum=None, optimum_label="Óptimo", xlabel="Episodio",
                   point_label=None, best_series=None):
    """Curva de aprendizaje con media móvil vs baseline y eval greedy.

    Args:
        best_series: opcional, mejor retorno POR BATCH (p. ej. la columna
            `max` del CSV). Se dibuja su máximo acumulado como línea naranja:
            el "mejor episodio encontrado hasta ahora", que visualiza cuándo
            la exploración alcanzó el óptimo aunque la media no lo muestre.
        baseline: array de retornos del baseline aleatorio; None omite su
            línea (útil al regenerar gráficos sin ese artefacto).
    """
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
    if best_series is not None:
        running_best = np.maximum.accumulate(
            np.asarray(best_series, dtype=np.float64))
        ax.plot(idx, running_best[idx], color="tab:orange", lw=1.4,
                label="Mejor episodio acumulado")
    if baseline is not None:
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
