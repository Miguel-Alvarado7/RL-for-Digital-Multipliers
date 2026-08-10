"""Artefactos por agente: TopK de circuitos, export Verilog, Q y returns."""

import os

import numpy as np

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


def save_returns(path, records):
    """Guarda `records` (lista de (episode, epsilon, return)) como CSV."""
    header = "episode,epsilon,return"
    data = np.array(records, dtype=float)
    np.savetxt(path, data, delimiter=",", header=header, comments="")
