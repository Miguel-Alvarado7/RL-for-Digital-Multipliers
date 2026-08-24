"""Resumen agregado de todos los agentes y configuraciones.

Recorre out/ buscando returns_cuda.csv, extrae metricas clave por corrida
y guarda un CSV consolidado en out/resumen_agentes.csv.

Sin reentrenar: las metricas de error (NMED, error_rate, peak) se derivan
del MEJOR circuito posible (best_multiplier_cuda.v) evaluado exhaustivamente
de forma bit-accurate (mismo comportamiento que iverilog+tb_multiplier_universal.v).

Columnas previas:
  algorithm, bits, n_steps, max_reward, min_reward, mean_reward, std_reward,
  last_reward, episodes_total
Nuevas (sin reentrenar, sobre mejor circuito):
  conv_ep       - episodios hasta primer max global (best circuito encontrado)
  conv_batch    - batch del conv_ep
  recompensa_promedio_final - media de los ultimos 100 batches (estable)
  recompensa_acumulada_total- suma aproximada de todos los retornos (mean*n_envs)
  mae           - Mean Absolute Error del MEJOR circuito sobre casos exhaustivos (verilog-accurate)
  nmed          - NMED = mae / (2^(2*bits)-1)
  error_rate    - % pares incorrectos = 100*(1-exact) del mejor
  peak_error    - max |P - A*B| del mejor
  terms         - nº wires pp !=0 del mejor (proxy area)
  height_inferido - height deducido de Q.shape (CC+1)/(2*bits) si existe
  verilog_src   - ruta del .v usado (best_multiplier_cuda.v o fallback Q greedy)
"""

import csv
import re
from pathlib import Path
import numpy as np

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
    "conv_ep",
    "conv_batch",
    "recompensa_promedio_final",
    "recompensa_acumulada_total",
    "mae",
    "nmed",
    "error_rate",
    "peak_error",
    "terms",
    "height_inferido",
    "verilog_src",
]

RX_BITS = re.compile(r"^bits(\d+)(?:_nsteps(\d+))?$")


def compute_conv_metrics(rows_csv):
    if not rows_csv:
        return None, None
    try:
        max_vals = [float(r[6]) for r in rows_csv if len(r) > 6]
        gmax = max(max_vals)
        idx = max_vals.index(gmax)
        conv_batch = int(float(rows_csv[idx][0]))
        conv_ep = int(float(rows_csv[idx][1]))
        return conv_ep, conv_batch
    except Exception:
        return None, None


def evaluate_verilog_file(verilog_path: Path):
    """Evalua un .v generado por BinaryMathEnv.generate_verilog() de forma
    bit-accurate (misma semantica que iverilog).

    Wire pp = 0/1 o (A[i]&B[j]) con negaciones; columna truncada a Bits bits;
    P truncada a 2*Bits bits. Exhaustivo sobre 4^Bits casos.

    Retorna dict con mae, peak, exact, nmed, error_rate, terms, bits.
    """
    text = verilog_path.read_text()
    m = re.search(r"input \[(\d+):0\] A", text)
    bits = int(m.group(1)) + 1 if m else 2

    pp_map = {}
    for line in text.splitlines():
        mm = re.search(r"wire pp(\d+)\s*=\s*(.+);", line)
        if mm:
            pp_map[f"pp{mm.group(1)}"] = mm.group(2).strip()
    col_map = {}
    for line in text.splitlines():
        mm = re.search(r"wire \[\d+:0\] (columna\d+)\s*=\s*(.+);", line)
        if mm:
            col_map[mm.group(1)] = mm.group(2).strip()
    mm = re.search(r"assign P\s*=\s*(.+);", text)
    assign_expr = mm.group(1) if mm else ""
    shifts = re.findall(r"\(?(columna\d+)\s*<<\s*(\d+)\)?", assign_expr)

    def eval_pp(expr, A, B):
        expr = expr.strip()
        if expr == "0":
            return 0
        if expr == "1":
            return 1
        expr = expr.strip("()")
        mm2 = re.match(r"(~?)A\[(\d+)\]&(~?)B\[(\d+)\]", expr)
        if mm2:
            neg_a = mm2.group(1) == "~"
            i = int(mm2.group(2))
            neg_b = mm2.group(3) == "~"
            j = int(mm2.group(4))
            a = (A >> i) & 1
            b = (B >> j) & 1
            if neg_a:
                a = 1 - a
            if neg_b:
                b = 1 - b
            return a & b
        return 0

    n = 1 << bits
    mask_col = (1 << bits) - 1
    mask_p = (1 << (2 * bits)) - 1
    denom_nmed = mask_p
    n_cases = n * n
    sum_err = 0
    peak = 0
    exact = 0
    # precompute terms = pp != "0"
    terms = sum(1 for v in pp_map.values() if v != "0")

    for A in range(n):
        for B in range(n):
            pp_vals = {k: eval_pp(v, A, B) for k, v in pp_map.items()}
            col_vals = {}
            for cname, expr in col_map.items():
                parts = [p.strip() for p in expr.split("+")]
                s = sum(pp_vals.get(p, 0) for p in parts)
                col_vals[cname] = s & mask_col
            P = 0
            for cname, sh in shifts:
                P += col_vals.get(cname, 0) << int(sh)
            P &= mask_p
            true = A * B
            err = abs(P - true)
            sum_err += err
            if err > peak:
                peak = err
            if err == 0:
                exact += 1

    mae = sum_err / n_cases if n_cases else None
    nmed = mae / denom_nmed if mae is not None and denom_nmed else None
    error_rate = (1 - exact / n_cases) * 100 if n_cases else None
    exact_frac = exact / n_cases if n_cases else None
    return dict(mae=mae, nmed=nmed, error_rate=error_rate, peak_error=float(peak),
                exact_frac=exact_frac, terms=float(terms), bits=bits, n_cases=n_cases)


def compute_greedy_fallback(bits, height, Q_path):
    """Fallback si no hay .v: evalua greedy de Q via cuda_optimized lineal (no mask)."""
    try:
        import torch
        from envs.cuda_optimized import BinaryMathEnvCUDAOptimized
        Q = np.load(Q_path)
        n_states = Q.shape[0]
        CC = n_states - 1
        greedy = Q[:CC].argmax(axis=1).astype(np.int64)
        env = BinaryMathEnvCUDAOptimized(Bits=bits, height=height, n_envs=1, device="cpu")
        grids = torch.tensor(greedy, dtype=torch.long).unsqueeze(0)
        mae_t, terms_t, exact_t, _ = env.true_metrics(grids)
        mae = float(mae_t.item())
        terms = float(terms_t.item())
        exact = float(exact_t.item())
        # peak via loop masked? usar linear peak (sin mascara) como fallback
        presence, _ = env._grid_presence(grids)
        peak = 0.0
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for lo in range(0, env.n_test_cases, env.chunk_size):
                hi = min(lo + env.chunk_size, env.n_test_cases)
                T = env._true_P[lo:hi]
                P = presence @ env._action_vals_for(lo, hi).T
                e = (P - T.unsqueeze(0)).abs()
                cur = float(e.max().item())
                if cur > peak:
                    peak = cur
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        nmed = mae / ((1 << (2 * bits)) - 1) if mae is not None else None
        error_rate = (1 - exact) * 100
        return dict(mae=mae, nmed=nmed, error_rate=error_rate, peak_error=peak, terms=terms)
    except Exception as e:
        print(f"  fallback greedy fallo bits{bits} h{height}: {e}")
        return dict(mae=None, nmed=None, error_rate=None, peak_error=None, terms=None)


def main():
    rows = []

    for csv_path in OUT_ROOT.rglob("returns_cuda.csv"):
        run_dir = csv_path.parent
        algorithm = run_dir.parent.name
        m = RX_BITS.match(run_dir.name)
        if m is None:
            continue
        bits = int(m.group(1))
        n_steps = int(m.group(2)) if m.group(2) else None

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            rows_csv = list(reader)

        if not rows_csv:
            continue

        max_vals = []
        min_vals = []
        mean_vals = []
        std_vals = []
        for row in rows_csv:
            if len(row) > 6:
                try: max_vals.append(float(row[6]))
                except: pass
            if len(row) > 5:
                try: min_vals.append(float(row[5]))
                except: pass
            if len(row) > 3:
                try: mean_vals.append(float(row[3]))
                except: pass
            if len(row) > 4:
                try: std_vals.append(float(row[4]))
                except: pass

        max_reward = max(max_vals) if max_vals else None
        min_reward = min(min_vals) if min_vals else None
        mean_reward = sum(mean_vals) / len(mean_vals) if mean_vals else None
        std_reward = sum(std_vals) / len(std_vals) if std_vals else None
        last_row = rows_csv[-1]
        last_reward = float(last_row[3]) if len(last_row) > 3 else None

        try:
            if len(rows_csv) >= 2:
                n_envs = int(float(rows_csv[1][1]) - float(rows_csv[0][1]))
            else:
                n_envs = 512
        except:
            n_envs = 512
        episodes_total = len(rows_csv) * n_envs

        conv_ep, conv_batch = compute_conv_metrics(rows_csv)

        try:
            last100 = mean_vals[-100:] if len(mean_vals) >= 100 else mean_vals
            recompensa_promedio_final = sum(last100) / len(last100) if last100 else None
        except:
            recompensa_promedio_final = None
        try:
            recompensa_acumulada_total = float(np.sum(mean_vals) * n_envs) if mean_vals else None
        except:
            recompensa_acumulada_total = None

        # --- mejor circuito ---
        verilog_src = None
        mae = nmed = error_rate = peak_error = terms = None
        height_inferido = None
        Q_path = run_dir / "Q_cuda.npy"
        if Q_path.exists():
            try:
                Q = np.load(Q_path)
                height_inferido = int((Q.shape[0] - 1) // (2 * bits)) if bits else None
            except:
                height_inferido = None

        best_v = run_dir / "best_multiplier_cuda.v"
        if best_v.exists():
            try:
                res = evaluate_verilog_file(best_v)
                mae = res["mae"]
                nmed = res["nmed"]
                error_rate = res["error_rate"]
                peak_error = res["peak_error"]
                terms = res["terms"]
                verilog_src = str(best_v).replace("\\", "/")
                # sanity: bits del .v debe coincidir con dir; si no, advertir
                if res["bits"] != bits:
                    print(f"[{algorithm} bits{bits}] warning: bits .v={res['bits']} != dir")
            except Exception as e:
                print(f"[{algorithm} bits{bits}] error evaluando {best_v}: {e}")
                # fallback a greedy
                if Q_path.exists() and height_inferido is not None:
                    fb = compute_greedy_fallback(bits, height_inferido, Q_path)
                    mae, nmed, error_rate, peak_error, terms = fb["mae"], fb["nmed"], fb["error_rate"], fb["peak_error"], fb["terms"]
                    verilog_src = f"fallback Q greedy ({Q_path})"
        else:
            # sin .v, fallback greedy
            if Q_path.exists() and height_inferido is not None:
                fb = compute_greedy_fallback(bits, height_inferido, Q_path)
                mae, nmed, error_rate, peak_error, terms = fb["mae"], fb["nmed"], fb["error_rate"], fb["peak_error"], fb["terms"]
                verilog_src = f"fallback Q greedy ({Q_path})"
            else:
                print(f"[{algorithm} bits{bits}] sin best .v ni Q")

        rows.append([
            algorithm, bits, n_steps, max_reward, min_reward, mean_reward, std_reward,
            last_reward, episodes_total,
            conv_ep, conv_batch, recompensa_promedio_final, recompensa_acumulada_total,
            mae, nmed, error_rate, peak_error, terms, height_inferido, verilog_src,
        ])

    rows.sort(key=lambda r: (r[0], r[1], r[2] if r[2] is not None else -1))

    with open(RESUME_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"CSV de resumen creado en {RESUME_FILE}")
    print(f"Filas procesadas: {len(rows)}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
