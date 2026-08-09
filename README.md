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
| **PyTorch** | Sí (train_mc_cuda.py) | En `requirements.txt`. Con GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124` (fallback automático a CPU si no hay GPU) |
| **pygame** | Opcional (solo render `BinaryMathEnvSecuencial`) | `pip install pygame` |
| **CUDA + driver NVIDIA** | Opcional (aceleración GPU) | Driver actualizado + PyTorch con soporte cu12x |

## Instalación rápida

```bash
pip install -r requirements.txt
pip install pygame          # opcional
# instalar Icarus Verilog según tu SO (tabla de arriba)
```

## Uso

```bash
# Entrenamiento CPU (evalúa con iverilog, un episodio a la vez)
python train_mc.py --bits 2 --episodes 1000

# Entrenamiento batched (GPU; usa CPU automáticamente si no hay CUDA)
python train_mc_cuda.py --n-envs 512 --episodes 100000

# Verificar el Verilog del mejor circuito encontrado (desde la raíz del repo)
iverilog -s test_multiplier_tb -o out/simv out/best_multiplier_cuda.v verification/test_multiplier_tb.v
vvp out/simv
```

Nota: los testbenches viven en `verification/` (versionados) y los circuitos
generados en `out/` (ignorados por git). El testbench de 2 bits es el que
verifica `best_multiplier_cuda.v` tras entrenar con `--bits 2`; para otros
`--bits` hay que generar un testbench equivalente (anchos `[N-1:0]` y barrido
hasta `2^N-1`).

## Estructura

- `Environment/Environment/` — entornos: `BinaryMathEnv` (CPU/iverilog), `BinaryMathEnvCUDA`, `BinaryMathEnvCUDAOptimized` (batched, streaming por chunks)
- `agents/` — `MonteCarloAgent` (CPU), `MonteCarloAgentCUDA` (batched, early-stop con confirmación greedy)
- `train_mc.py`, `train_mc_cuda.py` — scripts de entrenamiento
- `out/` — resultados (Q, returns, `best_multiplier*.v`)

## Flags útiles (train_mc_cuda.py)

- `--bits N` / `--height N` — tamaño del multiplicador y filas de la tabla
- `--n-envs N` / `--episodes N` — paralelismo y presupuesto de episodios
- `--device cuda|cpu` — forzar dispositivo (por defecto `cuda` con fallback)
- `--early-stop VAL` — umbral de parada (0.0 = circuito perfecto; `none` lo desactiva)
