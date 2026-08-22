# Multiplicadores Binarios con Reinforcement Learning

Síntesis de circuitos multiplicadores binarios mediante RL: un agente rellena
una tabla de productos parciales y el entorno la evalúa como un circuito
(Verilog real en CPU, tensores vectorizados en CUDA).

## Herramientas a instalar

| Herramienta | Necesaria | Instalación |
|---|---|---|
| **Python 3.10+** | Sí | python.org (añadir a PATH) |
| **pip packages** | Sí | `pip install -r requirements.txt` |
| **Icarus Verilog** (`iverilog`, `vvp`) | Sí (CPU y verificación) | Windows: `choco install iverilog` o instalador de bleyer.org · Ubuntu: `sudo apt install iverilog` · macOS: `brew install icarus-verilog` |
| **PyTorch** | Sí (scripts `*_cuda.py`) | En `requirements.txt`. Con GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124` (fallback automático a CPU si no hay GPU) |
| **pygame** | Opcional (solo render `BinaryMathEnvSecuencial`) | `pip install pygame` |
| **CUDA + driver NVIDIA** | Opcional (aceleración GPU) | Driver actualizado + PyTorch con soporte cu12x |

## Instalación rápida

```bash
pip install -r requirements.txt
pip install pygame          # opcional
# instalar Icarus Verilog según tu SO (tabla de arriba)
```

## Uso

Cada algoritmo tiene un script CPU (iverilog por episodio) y otro batched CUDA
(GPU; usa CPU automáticamente si no hay CUDA). Todos escriben en su propia
carpeta `out/<algo>/`.

```bash
# Entrenamiento CPU (evalúa con iverilog, un episodio a la vez)
python train_mc.py --bits 2 --episodes 1000
python train_qlearning.py --bits 2 --episodes 1000
python train_sarsa.py --bits 2 --episodes 1000

# Entrenamiento batched (GPU; usa CPU automáticamente si no hay CUDA)
python train_mc_cuda.py --n-envs 512 --episodes 100000
python train_qlearning_cuda.py --n-envs 512 --episodes 100000
python train_sarsa_cuda.py --n-envs 512 --episodes 100000

# Verificar el Verilog del mejor circuito encontrado (desde la raíz del repo).
# Un solo testbench universal sirve para cualquier --bits: pasa el ancho con -D.
iverilog -s tb_multiplier -D BITS=2 -o out/simv out/montecarlo_cuda/best_multiplier_cuda.v verification/tb_multiplier_universal.v
vvp out/simv

# Para otros anchos, cambia BITS y el DUT:
iverilog -s tb_multiplier -D BITS=6 -o out/simv out/montecarlo_cuda/bits6/best_multiplier_cuda.v verification/tb_multiplier_universal.v
vvp out/simv   # PASS/FAIL + MAE, Peak Error y % de respuestas exactas
```

Nota: el testbench universal y la plantilla viven en `verification/`
(versionados) y los circuitos generados en `out/<algo>/` (ignorados por git).
El testbench hace un barrido exhaustivo 2^BITS × 2^BITS comparando con A·B;
sin `-D BITS=<n>` compila por defecto a 2 bits. El número de productos
parciales no es visible desde la simulación: se cuenta con
`grep -c "wire pp" <dut>.v`.

## Algoritmos

| Algoritmo | Tipo | Update | Script CPU | Script CUDA |
|---|---|---|---|---|
| **Monte Carlo** | On-policy, first-visit | `Q += α(G − Q)` | `train_mc.py` | `train_mc_cuda.py` |
| **Q-learning** | Off-policy TD(0) | `Q += α(r + γ·max Q' − Q)` | `train_qlearning.py` | `train_qlearning_cuda.py` |
| **SARSA** | On-policy TD(0) | `Q += α(r + γ·Q' − Q)` | `train_sarsa.py` | `train_sarsa_cuda.py` |

- Escala de reward **CPU**: `100 − 100·error` (100 = circuito perfecto).
- Escala de reward **CUDA**: `[-10, 0]` (0 = circuito perfecto).
- Los agentes CUDA añaden `--gamma` (default `0.95`) para TD.
- Todos los updates CUDA promedian los targets por par `(s, a)` con
  `index_put_(accumulate=True)`: como n_envs episodios comparten los mismos
  estados, la indexación avanzada `Q[states, actions] += ...` descartaría
  todos los duplicados menos uno.

## Top 5 de circuitos

Todos los scripts guardan los **5 mejores circuitos** distintos vistos durante
el entrenamiento (ordenados por retorno, deduplicados por rejilla) en la carpeta
del agente:

- `out/montecarlo_cuda/best_multiplier_cuda.v` … `best_multiplier_cuda_5.v`
- `out/qlearning_cuda/best_multiplier_cuda.v` … `best_multiplier_cuda_5.v`
- `out/sarsa_cuda/best_multiplier_cuda.v` … `best_multiplier_cuda_5.v`
- (y análogo `best_multiplier.v` en `out/montecarlo/`, `out/qlearning/`, `out/sarsa/`)

Si hay menos de 5 circuitos distintos, se guardan los que haya.

## Estructura

- `envs/` — entornos: `base.py` (`BinaryMathEnv`, CPU/iverilog), `pygame_env.py` (`BinaryMathEnvSecuencial`), `cuda.py` (`BinaryMathEnvCUDA`), `cuda_optimized.py` (`BinaryMathEnvCUDAOptimized`, batched con streaming por chunks)
- `agents/` — base común (`_base.py`) y subpaquetes por motor:
  - `agents/cpu/` — `base.py` (tabular CPU), `montecarlo.py`, `qlearning.py`, `sarsa.py`
  - `agents/cuda/` — `base.py` (tabular batched), `montecarlo.py`, `qlearning.py`, `sarsa.py`
- `training/` — helpers compartidos: `baseline.py`, `plot.py`, `artifacts.py` (TopK + export Verilog)
- `verification/` — testbench universal (`tb_multiplier_universal.v`, parametrizado con `-D BITS=<n>`) y plantilla del entorno CPU (`testbench_template.v`)
- `docs/environment.md` — documentación exhaustiva de los entornos
- `out/` — resultados por agente (Q, returns, `best_multiplier*.v`, plots) y scratch de Verilog (`out/verilog/`)

## Cómo evalúa el entorno CUDA optimized

`BinaryMathEnvCUDAOptimized` soluciona el OOM de `BinaryMathEnvCUDA` eliminando
la expansión masiva de pares `(A, B)` a GPU:

1. **Cachea `action_vals`** — el valor de cada acción solo depende de los pares
   exhaustivos y del espacio de acciones, NO del entorno ni de `n_envs`, así
   que se calcula una vez (si no cabe en memoria, se recalcula por chunk).
2. **Evalúa cada rejilla con un único GEMM** en float32 (con TF32 desactivado
   por exactitud), con los factores de desplazamiento plegados en la matriz de
   presencia: `O(n_envs·n_actions·chunk)` en lugar de iterar por celdas.
3. **Procesa los casos de prueba en chunks** (`chunk_size` auto-ajustado por
   presupuesto de memoria) y expone `commit_episodes`/`evaluate_grids` para
   evaluar episodios completos o rejillas arbitrarias de una sola vez.

Medido en una RTX 4000 Ada (batch de entrenamiento completo):

```
Bits=8, n_envs=64     627 ms -> 0.93 ms   (1875 MB -> 122 MB)
Bits=8, n_envs=1024      OOM -> 13.5 ms   (846 MB)
Bits=4, n_envs=4096    51.9 ms -> 0.93 ms (1616 MB -> 30 MB)
```

## Flags útiles (scripts CUDA)

- `--bits N` / `--height N` — tamaño del multiplicador y filas de la tabla
- `--n-envs N` / `--episodes N` — paralelismo y presupuesto de episodios.
  Un batch es un único GEMM, así que `--n-envs` bajo no acelera nada (con 64 o
  con 4096 el batch tarda lo mismo); lo que gobierna el aprendizaje es el
  número de batches = `episodes / n_envs`. En GPU se recomienda ~1024; el
  default del script es 512.
- `--baseline-batches N` — batches de la referencia aleatoria (default 8)
- `--alpha A` / `--gamma G` — tasa de aprendizaje y descuento (TD)
- `--device cuda|cpu` — forzar dispositivo (por defecto `cuda` con fallback)
- `--early-stop VAL` — umbral de parada (0.0 = circuito perfecto; `none` lo desactiva)
