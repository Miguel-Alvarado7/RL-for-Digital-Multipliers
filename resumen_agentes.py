"""Resumen agregado de todos los agentes y configuraciones.

Recorre out/ buscando returns_cuda.csv, extrae metricas clave por corrida
y guarda un CSV consolidado en out/resumen_agentes.csv.

Columnas:
  algorithm  - Nombre del agente (montecarlo_cuda, qlearning_cuda, sarsa_cuda)
  bits       - Precision del multiplicador (2, 4, 6)
  n_steps    - Pasos de lookahead (None para MC/Q, 1/20/30 para SARSA)
  max_reward - Mejor retorno por batch (columna max del CSV)
  min_reward - Peor retorno por batch (columna min del CSV)
  mean_reward- Media de retornos por batch (columna mean)
  std_reward - Desviacion estandar de retornos por batch (columna std)
  last_reward- Retorno de la ultima corrida (columna mean de la ultima fila)
  episodes_total - Total de episodios simulados (50M fijo)
"""

import csv
import re
from pathlib import Path

OUT_ROOT = Path("out")
RESUME_FILE = OUT_ROOT / "resumen_agentes.csv"

HEADER = [
    "algorithm",
    "bits",
    "n_steps",
    "max_reward",
    "min_reward",
    "mean_reward",
    "std_reward",
    "last_reward",
    "episodes_total",
]

# Patrones de nombre de directorio.
# El nombre del directorio padre (run_dir.parent.name) es el algorithm.
# El nombre del directorio actual (run_dir.name) contiene bits y opcionalmente _nsteps.
RX_BITS = re.compile(r"^bits(\d+)(?:_nsteps(\d+))?$")


def parse_dir_name(dir_name: str):
    """Devuelve (algorithm, bits, n_steps) o None si no coincide el patron."""
    m = RX_DIR.match(dir_name)
    if not m:
        return None
    algorithm = m.group(1)
    bits = int(m.group(2))
    n_steps = int(m.group(3)) if m.group(3) else None
    return algorithm, bits, n_steps


def main():
    rows = []

    # Buscar todos los returns_cuda.csv de forma recursiva bajo out/
    for csv_path in OUT_ROOT.rglob("returns_cuda.csv"):
        run_dir = csv_path.parent

        # El algorithm viene del directorio padre (ej. montecarlo_cuda, qlearning_cuda, sarsa_cuda)
        algorithm = run_dir.parent.name

        # Extraer bits y n_steps del nombre del directorio actual (ej. bits2, bits4, bits2_nsteps1)
        m = RX_BITS.match(run_dir.name)
        if m is None:
            # Directorios que no siguen el patron se omiten (verilog, simv, etc.)
            continue
        bits = int(m.group(1))
        n_steps = int(m.group(2)) if m.group(2) else None

        # Leer CSV (formato por batch: batch,episode_start,epsilon,mean,std,min,max,p25,p50,p75)
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # saltar header
            rows_csv = list(reader)

        if not rows_csv:
            continue  # CSV vacio, skip

        # Calcular max/min/mean/std sobre la columna correspondiente (índice 6=max, 5=min, 3=mean, 4=std)
        # sobre todas las filas (todas las batches), no solo la ultima.
        max_vals = []
        min_vals = []
        mean_vals = []
        std_vals = []
        for row in rows_csv:
            if len(row) > 6:
                max_vals.append(float(row[6]))
            if len(row) > 5:
                min_vals.append(float(row[5]))
            if len(row) > 3:
                mean_vals.append(float(row[3]))
            if len(row) > 4:
                std_vals.append(float(row[4]))

        max_reward = max(max_vals) if max_vals else None
        min_reward = min(min_vals) if min_vals else None
        # mean_reward: tomamos el mean promedio sobre all batches
        mean_reward = sum(mean_vals) / len(mean_vals) if mean_vals else None
        std_reward = sum(std_vals) / len(std_vals) if std_vals else None

        # last_reward: retention of the last batch's mean (útil para seguimiento)
        last_row = rows_csv[-1]
        last_reward = float(last_row[3]) if len(last_row) > 3 else None

        rows.append([
            algorithm,
            bits,
            n_steps,
            max_reward,
            min_reward,
            mean_reward,
            std_reward,
            last_reward,
            50_000_000,  # episodes_total fijo segun la configuracion
        ])

    # Escribir CSV consolidado
    with open(RESUME_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"CSV de resumen creado en {RESUME_FILE}")
    print(f"Filas procesadas: {len(rows)}")


if __name__ == "__main__":
    main()