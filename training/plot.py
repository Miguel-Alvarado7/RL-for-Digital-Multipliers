"""Curva de aprendizaje genérica (media móvil vs baseline y eval greedy)."""

import numpy as np


def learning_curve(returns, baseline, eval_mean, path, title,
                   ylabel="Retorno (reward)", series_label="Agente",
                   optimum=None, optimum_label="Óptimo", xlabel="Episodio",
                   point_label=None, best_series=None, max_series=None):
    """Curva de aprendizaje con media móvil vs baseline y eval greedy.

    Args:
        best_series: opcional, mejor retorno POR BATCH (p. ej. la columna
            `max` del CSV). Se dibuja su media móvil como línea neta (ancho 2,
            alpha 0.7) en lugar del máximo acumulado: visualiza cuándo la
            exploración alcanzó el óptimo aunque la media no lo muestre.
        max_series: opcional, columna `max` del CSV trazada como línea naranja
            transparente (alpha 0.2, ancho 1.5) de episodios por batch, estilo
            sólido (sin punteado). Si no se provee, se omite.
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
        window = max(10, len(returns) // 50)
        csum = np.concatenate([[0.0], np.cumsum(best_series, dtype=np.float64)])
        smooth_best = (csum[window:] - csum[:-window]) / window
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
