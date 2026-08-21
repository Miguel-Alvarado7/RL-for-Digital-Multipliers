"""Artefactos por agente: TopK de circuitos, export Verilog, Q y returns."""

import os

import numpy as np
import torch

from envs import BinaryMathEnv


class TopK:
    """Mantiene los k mejores circuitos (dedup por rejilla, ordenados por retorno).

    La rejilla (key) identifica de forma única al circuito. Si un mismo key
    vuelve con mejor retorno, se actualiza.
    """

    def __init__(self, k=5):
        self.k = k
        self.items = []  # list[(return, key)], desc por return

    def add(self, return_, key):
        for i, (r, kk) in enumerate(self.items):
            if kk == key:
                if return_ > r:
                    self.items[i] = (return_, key)
                    self.items.sort(key=lambda x: x[0], reverse=True)
                return
        self.items.append((return_, key))
        self.items.sort(key=lambda x: x[0], reverse=True)
        if len(self.items) > self.k:
            self.items = self.items[:self.k]

    def __len__(self):
        return len(self.items)


def topk_add_batch(topk, returns, grids):
    """Alimenta `topk` con las mejores rejillas DISTINTAS de un batch.

    Equivalente exacto a recorrer los n_envs entornos uno a uno, pero sin el
    bucle de Python: aquel hacia una transferencia GPU->CPU por entorno y
    reordenaba la lista en cada `add`, lo que costaba ~15 ms/batch frente a
    ~1 ms de computo real en GPU (la GPU quedaba al 25%).

    Por que basta con las k mejores distintas: TopK deduplica por rejilla, y
    una rejilla que no esta entre las k mejores DISTINTAS del batch tiene ya k
    rejillas distintas de este mismo batch por encima, asi que no puede entrar
    en un top-k global. La deduplicacion se hace en GPU para que la seleccion
    sea sobre rejillas distintas y no sobre repetidas (con epsilon bajo muchos
    entornos producen la misma rejilla).

    Args:
        topk:    Instancia de TopK.
        returns: (n_envs,) tensor de retornos.
        grids:   (n_envs, CC) tensor de rejillas.
    """
    uniq, inverse = torch.unique(grids, dim=0, return_inverse=True)
    best = torch.full((uniq.shape[0],), float('-inf'),
                      dtype=returns.dtype, device=returns.device)
    best.scatter_reduce_(0, inverse, returns, reduce='amax')

    k = min(topk.k, uniq.shape[0])
    vals, idx = best.topk(k)
    for r, row in zip(vals.cpu().tolist(), uniq[idx].cpu().tolist()):
        topk.add(r, tuple(row))


def write_verilog(grid_list, bits, height, path):
    """Escribe el Verilog de una rejilla de términos (str) en `path`."""
    env = BinaryMathEnv(Bits=bits, Proof=8, height=height)
    env.suma_grid = list(grid_list)
    env.generate_verilog()
    with open(path, "w") as f:
        f.write(env.last_verilog_code)
    return path


def save_topk(topk, out_dir, bits, height, base_name, grid_from_key):
    """Guarda los circuitos del TopK como Verilog en `out_dir`.

    Args:
        topk:           TopK ya poblado.
        out_dir:        carpeta de salida del agente (se crea si falta).
        bits, height:   configuración del multiplicador.
        base_name:      sin extensión; el rank 1 se llama {base_name}.v y los
                        demás {base_name}_{rank}.v.
        grid_from_key:  callable key -> lista de términos (str) de la rejilla.

    Returns:
        list[str]: rutas guardadas.
    """
    if not topk.items:
        print("No se encontraron circuitos (sin episodios ejecutados).")
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    print(f"Top {len(topk.items)} circuitos (retorno -> archivo):")
    for rank, (ret, key) in enumerate(topk.items, start=1):
        name = f"{base_name}.v" if rank == 1 else f"{base_name}_{rank}.v"
        path = write_verilog(grid_from_key(key), bits, height,
                             os.path.join(out_dir, name))
        saved.append(path)
        print(f"  #{rank}  return={ret:8.3f}  {path}")
    return saved


def save_q(path, q):
    """Guarda la tabla Q (numpy o torch, CPU/GPU) en `path`."""
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    np.save(path, np.asarray(q))


def save_returns(path, records, n_envs=None, per_episode=False):
    """Guarda `records` (lista de (episode, epsilon, return)) como CSV.

    Por defecto agrega POR BATCH en vez de escribir una fila por episodio.
    Los n_envs episodios de un batch son muestras i.i.d. de la misma politica
    con el mismo epsilon (verificado: epsilon es constante dentro de cada
    batch), asi que el indice de episodio dentro del batch es orden de entorno
    en el tensor, no orden temporal: guardar por episodio no aporta resolucion
    temporal, solo tamano. Con 10M episodios son 725 MB frente a ~2 MB, y la
    curva de aprendizaje reconstruida desde las medias por batch difiere de la
    calculada sobre todos los episodios en 2.4e-2 como maximo (sobre un reward
    en [-10, 0]).

    Se conservan std y percentiles para no perder la dispersion intra-batch.

    Args:
        path:        Destino del CSV.
        records:     Lista de (episodio_global, epsilon, retorno).
        n_envs:      Episodios por batch. Si es None se escribe por episodio.
        per_episode: True fuerza el formato antiguo (una fila por episodio).
    """
    data = np.array(records, dtype=float)
    if per_episode or not n_envs:
        header = "episode,epsilon,return"
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        return

    n_batches = len(data) // n_envs
    if n_batches == 0:                      # menos de un batch completo
        header = "episode,epsilon,return"
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        return

    trimmed = data[:n_batches * n_envs]
    ret = trimmed[:, 2].reshape(n_batches, n_envs)
    eps = trimmed[:, 1].reshape(n_batches, n_envs)[:, 0]
    p25, p50, p75 = np.percentile(ret, [25, 50, 75], axis=1)

    out = np.column_stack([
        np.arange(n_batches),               # batch
        np.arange(n_batches) * n_envs,      # episodio inicial del batch
        eps,
        ret.mean(axis=1), ret.std(axis=1),
        ret.min(axis=1), ret.max(axis=1),
        p25, p50, p75,
    ])
    header = ("batch,episode_start,epsilon,mean,std,min,max,p25,p50,p75")
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.6g")
