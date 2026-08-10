# Entorno de Aprendizaje por Refuerzo para Multiplicadores Binarios

Documentación exhaustiva del paquete `envs`.

Este documento describe en detalle el **entorno de Aprendizaje por Refuerzo (RL)** que
sintetiza circuitos multiplicadores binarios. Cubre las cuatro clases del paquete, el
pipeline de evaluación Verilog real, las fórmulas de recompensa y las limitaciones
conocidas. **No** cubre agentes, algoritmos de RL ni entrenamiento.

---

## 1. Resumen del problema

El entorno modela un problema de **diseño de circuitos digitales mediante RL**:

- **Objetivo**: construir un *multiplicador binario* de `Bits × Bits` que produzca el
  producto correcto `P = A × B` para todos los operandos posibles, minimizando además la
  complejidad del circuito.
- **Representación del diseño**: el circuito se representa como una **tabla de productos
  parciales** de `height` filas × `2·Bits` columnas. Cada celda debe contener un término
  booleano de la forma:

  ```
  (A[i]&B[j])   (~A[i]&B[j])   (A[i]&~B[j])   (~A[i]&~B[j])   0    1
  ```

  Cada columna de la tabla acumula la **suma** de los términos que contiene; las sumas de
  las columnas se desplazan y se suman para producir `P` (exactamente como lo haría un
  multiplicador de suma de productos parciales en hardware).

- **El agente**: rellena la tabla celda por celda (en orden), eligiendo qué término
  booleano colocar en cada casilla. Cuando la tabla está completa, el entorno la
  **compila a Verilog**, la **simula** y otorga una recompensa según el error funcional
  medido contra el producto esperado.

El espacio de diseño crece exponencialmente con `Bits`: es un problema combinatorio que
motiva el uso de RL (y de búsqueda tipo MCTS) para explorar tablas válidas.

---

## 2. Arquitectura del paquete

```
envs/
├── __init__.py          Re-exporta las 4 clases (imports perezosos)
├── base.py              BinaryMathEnv          — núcleo conceptual (gym.Env)
├── pygame_env.py        BinaryMathEnvSecuencial — versión con render pygame
├── cuda.py              BinaryMathEnvCUDA       — batch vectorizado en GPU
└── cuda_optimized.py    BinaryMathEnvCUDAOptimized — batch con streaming por chunks
```

| Clase | Motor de evaluación | Interfaz | Uso típico |
|---|---|---|---|
| `BinaryMathEnv` | `iverilog` + `vvp` (Verilog real) | Gym clásico (`reset/step`) | Demos, validación conceptual, CPU |
| `BinaryMathEnvSecuencial` | `iverilog` + `vvp` | Gym + `render()` (pygame) | Visualización del proceso |
| `BinaryMathEnvCUDA` | Tensores PyTorch (GPU) | Batched (`step(actions[n_envs])`) | MCTS paralelo, escalado |
| `BinaryMathEnvCUDAOptimized` | Tensores PyTorch (chunks) | Batched | MCTS con `Bits` grandes (evita OOM) |

### 2.1 Imports perezosos (`__init__.py`)

El `__init__.py` re-exporta las cuatro clases, pero usa **imports perezosos (PEP 562)**
mediante un `__getattr__` a nivel de módulo. Solo `BinaryMathEnv` (que depende de
`numpy` y `gymnasium`) se importa de forma inmediata; las clases que arrastran
dependencias pesadas (`pygame`, `torch`) se cargan bajo demanda:

```python
_LAZY_MODULES = {
    'BinaryMathEnvSecuencial': '.pygame_env',
    'BinaryMathEnvCUDA': '.cuda',
    'BinaryMathEnvCUDAOptimized': '.cuda_optimized',
}

def __getattr__(name):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(...)
```

Esto permite ejecutar experimentos en CPU con **solo `numpy` + `gymnasium`** sin instalar
`pygame` ni `torch`. `__all__` sigue listando las cuatro clases, por lo que
`from envs import *` continúa funcionando.

---

## 3. `BinaryMathEnv` — el entorno CPU (núcleo conceptual)

Clase base del paquete. Hereda de `gymnasium.Env`. Toda la lógica conceptual del problema
vive aquí; las demás clases la replican o la aceleran.

### 3.1 Parámetros de configuración

```python
def __init__(self, Bits=8, Proof=4, height=8, maxi=100):
```

| Parámetro | Valor por defecto | Significado |
|---|---|---|
| `Bits` | 8 | Ancho de los operandos `A` y `B` en bits. El producto `P` tiene `2·Bits` bits. |
| `Proof` | 4 | Número de casos de prueba (pares `(A, B)`) usados en la simulación Verilog. |
| `height` | 8 | Filas de la tabla de productos parciales. |
| `maxi` | 100 | Límite de iteraciones (actualmente sin uso activo en la dinámica). |

**Derivados de configuración:**

- `grid_size = 2 * Bits` — número de columnas de la tabla.
- `CC = height * grid_size = height * 2 * Bits` — **total de casillas** de la tabla
  (= longitud del episodio).
- `suma_grid = [' '] * CC` — vector que representa la tabla (un string por celda;
  `' '` = vacía). Internamente es una lista plana indexada `row * grid_size + col`.
- `cursor_position = 0` — siguiente casilla a rellenar.
- `current_phase = 2`, `phase_names = ("Rellenar Tabla de Productos Parciales",)` —
  el entorno opera en una única fase (los nombres sugieren fases futuras de una
  metodología más amplia de diseño).
- `min_error = 0.9` — umbral para archivar la mejor solución en `multipliermax.v`.

### 3.2 Espacio de acciones

El agente debe elegir, para cada celda, **qué término booleano colocar**. Las acciones
posibles se generan en `_generate_possible_actions()`:

```
['0', '1']  +  para cada (i, j) en Bits×Bits:
    '(A[i]&B[j])'   '(~A[i]&B[j])'   '(A[i]&~B[j])'   '(~A[i]&~B[j])'
```

Por tanto:

```
n_actions = 2 + 4·Bits²
```

| `Bits` | `n_actions` | Tamaño de la tabla `CC` (height=Bits) |
|---|---|---|
| 2 | 18 | 8 |
| 4 | 66 | 32 |
| 8 | 258 | 128 |

```python
self.action_space = gym.spaces.Discrete(len(self.possible_actions))
```

### 3.3 Espacio de observación

```python
self.observation_space = gym.spaces.MultiDiscrete([self.CC, self.CC + 1])
```

La observación `_get_observation()` es un vector de 2 componentes:

```python
casillas_rellenas = sum(1 for cell in self.suma_grid if cell != ' ')
return np.array([self.cursor_position, casillas_rellenas], dtype=np.int32)
```

- `cursor_position` — índice de la próxima casilla a rellenar (0..CC).
- `casillas_rellenas` — número de celdas ya rellenadas.

**Observación efectiva.** Cada paso válido llena exactamente una celda y avanza el cursor
en uno, por lo que `cursor_position == casillas_rellenas` en todo momento. El estado
efectivo es, en la práctica, **un escalar: el cursor (0..CC)**, es decir, solo `CC+1`
estados distintos.

**⚠️ Advertencia (no-Markovianidad).** La observación **no contiene el contenido de la
rejilla**. Dos trayectorias distintas que llegan al mismo cursor con tablas diferentes son
indistinguibles. La recompensa terminal depende de *toda* la tabla, así que el proceso
de decisión no es Markoviano bajo esta observación. Los métodos tabulares aprenden una
heurística (no la política óptima garantizada). Para un tratamiento exacto habría que
incluir la rejilla en el estado (espacio astronómico) o usar aproximadores de función.

### 3.4 Semántica de `step`

```python
def step(self, action, arch_multiplier=None, arch_multipliermax=None,
         arch_multiplier_8bit_tb=None, arch_simv=None):
```

1. `reward = 0.0` y `CP += 1` (contador de pasos).
2. Validaciones:
   - `action` fuera de `[0, n_actions)` → `truncated = True` (episodio truncado).
   - `cursor_position >= CC` → episodio ya completado → `truncated = True`.
3. En caso normal: `suma_grid[cursor] = possible_actions[action]` y `cursor += 1`.
4. Si al llenar la celda el cursor alcanza `CC`, el episodio **termina**:
   - `it += 1` (contador de operaciones completadas).
   - `terminated = closed(...)` — se evalúa la tabla y se fija `self.reward` final.

Retorna la tupla Gym estándar: `(observation, reward, terminated, truncated, info)`.

**Nota sobre el episodio:** con acciones válidas la longitud es *fija*: exactamente `CC`
pasos. Las recompensas intermedias son siempre `0`; toda la señal llega en el último paso.

### 3.5 `reset`

Restablece `cursor_position = 0`, `suma_grid = [' '] * CC`, `it = CP = reward = 0`.
Retorna `(observación, {})`. Acepta `seed` (Gym estándar).

### 3.6 Utilidades de estado

- `clone()` — copia profunda (`copy.deepcopy`). Pensado para MCTS (explorar caminos
  sin alterar el entorno original).
- `get_state()` — serializa el estado a un dict (`cursor_position`, `suma_grid`, `it`,
  `CP`, `reward`, `min_error`, configuración, y opcionalmente `last_metrics` y
  `last_verilog_code`).
- `set_state(state)` — restaura el estado desde ese dict.

---

## 4. Pipeline de evaluación real (`generate_verilog` + `closed`)

La evaluación es el corazón del entorno: transforma la tabla rellena en un circuito
Verilog, lo compila y simula con Icarus Verilog, y calcula la recompensa.

```
tabla (suma_grid)  ──►  multiplier.v  +  multiplier_8bit_tb.v
                                  │
                    iverilog -o simv multiplier.v tb.v
                                  │
                        vvp simv  →  salida numérica
                                  │
                  error vs P_esperado  →  reward
```

### 4.1 De la tabla al Verilog (`generate_verilog`)

1. **Remodelado**: `suma_grid.reshape(height, 2·Bits)`.
2. **Deduplicación global**: se recorre la tabla y se conserva la **primera aparición**
   de cada término único (en orden de aparición). Así, dos celdas con el mismo producto no
   generan dos wires redundantes.
3. **Wires de productos**: cada término único produce un wire:
   ```
   wire pp0 = 0;
   wire pp1 = (A[1]&B[1]);
   ...
   ```
4. **Sumas por columna**: para cada columna `col` (0..2·Bits−1), se toman los términos
   únicos presentes, se suman y se asignan a `columna{idx}` donde:

   ```
   idx = 2·Bits − col
   ```

   ```
   wire [Bits-1:0] columna4 = pp0;
   wire [Bits-1:0] columna3 = pp1 + pp2;
   ...
   ```
5. **Asignación de salida**: se suman las columnas desplazadas:

   ```
   assign P = (columna{i1} << (i1−1)) + (columna{i2} << (i2−1)) + ...
   ```

   **Mapeo columna → bit.** La columna física `col` de la tabla contribuye al bit
   `2·Bits − col − 1` de `P`. Es decir, la columna `col = 2·Bits − 1` (la más a la
   derecha) es el **LSB** (bit 0) y la columna `col = 0` es el **MSB**. Este mapeo se
   replica en la versión CUDA mediante `shift_factors = 2^(grid_size − col − 1)`.

   *Ejemplo* (Bits=2, P de 4 bits):

   ```
   col = 0 → columna4 << 3   (bit 3)
   col = 1 → columna3 << 2   (bit 2)
   col = 2 → columna2 << 1   (bit 1)
   col = 3 → columna1 << 0   (bit 0)
   ```

6. El código se escribe en `out/verilog/multiplier.v` y se almacena en
   `last_verilog_code` (para visualización).

### 4.2 Generación del testbench

Se lee la plantilla `verification/testbench_template.v` y se sustituyen los
placeholders:

| Placeholder | Sustitución |
|---|---|
| `{regsI}` | `Bits − 1` (ancho de `A` y `B`) |
| `{regsO}` | `2·Bits − 1` (ancho de `P`) |
| `{Test}` | un caso por cada par de prueba: `A = 8'd…; B = 8'd…; #10; $display("%d", P);` |

La plantilla instancia el módulo `multiplier` con puertos `A`, `B`, `P`, declara los
registros/wires de los anchos correctos, ejecuta el bloque de test y termina con
`$finish`. El testbench se escribe en `multiplier_8bit_tb.v`.

**Casos de prueba (CPU).** Se generan aleatoriamente:

```python
test_cases = np.random.randint(1, 2**Bits, size=(self.Proof, 2))
```

Valores en `[1, 2^Bits − 1]` (se excluye el 0, evitando divisiones por cero en el error).
El número de casos es `Proof` (4 por defecto). **Consecuencia:** la recompensa es
**estocástica** — la misma tabla puede recibir recompensas distintas en episodios
distintos según los casos muestreados.

### 4.3 Compilación y simulación

- `iverilog -o simv multiplier.v multiplier_8bit_tb.v` (compilación, timeout 10 s).
- `vvp simv` (simulación, timeout 10 s).
- Se parsea la salida estándar: todas las líneas que son enteros puros se interpretan
  como los valores de `P` emitidos por `$display`.

**Manejo de fallos.** Si la compilación falla, la simulación expira o la salida está
vacía, se devuelven arrays vacíos (los errores de compilación Verilog son esperables en
tablas aleatorias y se silencian). El llamador (`closed`) traduce esto a una penalización.

### 4.4 Cálculo de recompensa (`closed`)

Cuando el episodio termina:

1. Se obtienen `test_cases` y `results` (los `P` simulados).
2. Si no hay resultados válidos → `reward = -50` (y se termina).
3. Se calcula el error relativo normalizado:

   ```
   error_m = |P_sim − P_verdadero| / (P_verdadero + 1e-9)     para cada caso
   error_mean = mean(error_m)
   error_penalty = error_mean · 100
   ```

4. Se calculan métricas de circuito (`_calculate_circuit_metrics`):

   | Métrica | Definición |
   |---|---|
   | `logic_gates` | `code.count('&') + code.count('|')` |
   | `wires` | `code.count('wire')` |
   | `operand_count` | número de términos distintos en la tabla |
   | `complexity_penalty` | `10` si `logic_gates > 2·Bits²·2`, si no `0` |

5. **Recompensa final (CPU):**

   ```
   reward = 100 − error_penalty = 100 − 100·mean( |P_sim − P| / (P + 1e-9) )
   ```

   - Circuito perfecto → `error = 0` → **reward = 100**.
   - Circuitos con errores reciben recompensas decrecientes (pueden ser muy negativas,
     del orden de `−100·(error_mean)`).
   - Fallo de simulación → **reward = −50**.

6. **Archivo de la mejor solución**: si `error_mean < min_error`, se copia
   `multiplier.v` a `multipliermax.v` y se actualiza `min_error`. Es el mecanismo del
   entorno para guardar el mejor circuito encontrado hasta el momento.

7. Se guarda `last_metrics` (dict con `error_mean`, `error_penalty`, `circuit_metrics`,
   `final_reward`) para diagnóstico y visualización.

---

## 5. `BinaryMathEnvSecuencial` — renderizado (pygame)

Hereda de `BinaryMathEnv` y añade visualización. No cambia la dinámica del entorno.

- `render_mode` ∈ {`None`, `'human'`, `'rgb_array'`}.
- `render()` → `_render_frame()` (ventana pygame, 1200×700) o array RGB.
- **Panel izquierdo**: la tabla de productos parciales.
  - Casillas: `height × (2·Bits)` celdas de 60×25 px.
  - La celda del cursor se pinta con borde rojo y relleno amarillo.
  - Las celdas rellenas con borde azul y relleno verde.
  - Debajo: cursor (`cursor/CC`), casillas rellenas, recompensa actual.
- **Panel derecho**: información del circuito — número de productos únicos (lista de los
  primeros 5), y si hay `last_metrics`: error funcional, puertas lógicas, conexiones,
  operandos y recompensa final.
- **Vista previa Verilog**: primeras 8 líneas de `last_verilog_code`.

**Manejo de recursos pygame.** Los objetos `pygame.font.Font`, `window` y `clock` no se
pueden serializar con `copy.deepcopy`, así que:

- `clone()` guarda las referencias, las pone a `None`, hace `deepcopy` del resto y luego
  **reinicializa** fuentes/ventana en el clon.
- `set_state()` restaura el estado y reinicializa los recursos de render.
- `get_state()` añade `render_mode` al dict del padre.

---

## 6. `BinaryMathEnvCUDA` — entorno batched en GPU

Versión vectorizada que reemplaza la compilación Verilog (iverilog/vvp, ~100 ms por
episodio) por **operaciones tensoriales PyTorch**, evaluando miles de entornos en
paralelo. Mantiene el **mismo espacio de acciones** que `BinaryMathEnv` para ser
intercambiable en MCTS.

```python
BinaryMathEnvCUDA(Bits=8, Proof=4, height=8, n_envs=1024, device='cuda',
                  incremental=False, allow_negation=True, no_repeat=False)
```

### 6.1 Estado como tensores

| Tensor | Forma | Descripción |
|---|---|---|
| `suma_grid` | `(n_envs, CC)` int16 | índice de acción por celda; `−1` = vacía |
| `cursor_pos` | `(n_envs,)` int32 | próxima celda de cada entorno |
| `done` | `(n_envs,)` bool | episodio terminado |
| `rewards` | `(n_envs,)` float32 | recompensa acumulada |
| `carry_in` | `(n_envs, n_test_cases)` int32 | solo modo incremental |

### 6.2 Espacio de acciones parametrizado

- **`allow_negation=False`**: se omiten los productos negados; quedan `0, 1` +
  `(A[i]&B[j])` → `n_actions = 2 + Bits²`.
- **`no_repeat=True`**: cada producto parcial (índice ≥ 2) puede usarse **una sola vez**
  por episodio; las constantes `0`/`1` (`const_action_idx = [0,1]`) quedan exentas.
  El entorno mantiene `used_actions (n_envs, n_actions)` y expone
  `get_valid_action_mask()` / `get_valid_actions()` / `sample_valid_actions()`; el agente
  debe muestrear solo acciones válidas (action-masking).

### 6.3 Test cases exhaustivos (deterministas)

A diferencia del CPU, se generan **todos** los pares `(A, B)` con
`A, B ∈ [0, 2^Bits − 1]`:

```
n_test_cases = (2^Bits)²            (Bits=2 → 16, Bits=4 → 256, Bits=8 → 65536)
max_product  = (2^Bits − 1)²
```

Los tensores `A_vals`, `B_vals`, `true_products` son idénticos para todos los entornos
(`unsqueeze(0).expand(n_envs, −1)`). Esto hace la **evaluación determinista**: la misma
tabla produce siempre la misma recompensa (no hay muestreo aleatorio de casos).

### 6.4 Tablas de decodificación precomputadas

Para cada acción parcial se precomputan en GPU: `action_i_bits`, `action_j_bits` (qué
bit de `A`/`B` usa), `action_neg_a`, `action_neg_b` (si está negado) y
`shift_factors = 2^(grid_size − col − 1)` (el desplazamiento Verilog por columna).

### 6.5 API batched

- `reset(env_indices=None)` — reinicia todos o un subconjunto.
- `step(actions)` — `actions` es un tensor `(n_envs,)`. Rellena la celda del cursor de
  cada entorno activo, avanza el cursor y detecta finales.
- `rollout_from_state(state_dict, n_rollouts, rollout_depth)` — ejecuta N rollouts
  aleatorios paralelos desde un estado de MCTS (carga el estado con `_load_state`).
- `clone_env(src, dst)` — clona un entorno dentro del batch (in-place, sin copias CPU).
- `get_single_state(env_idx)` / `_load_state(state)` — interoperabilidad con
  `BinaryMathEnv` (convierte strings ↔ índices).

### 6.6 Evaluación — modo original (reward al final del episodio)

`_compute_products` replica **exactamente** la lógica Verilog:

1. **One-hot** de la rejilla → `(n, height, grid_size, n_actions)`.
2. **Deduplicación** por columna: `max` sobre `height` actúa como OR lógico — si un
   producto aparece ≥ 1 vez en la columna, contribuye una vez.
3. `col_sums = einsum('eca,epa->ecp')` — suma ponderada de valores de acciones por
   columna.
4. `products = einsum('ecp,c->ep')` con `shift_factors` — desplazamientos y suma total.

Recompensa (error ponderado por bits con riesgo exponencial):

```
P_calc, error = |P_calc − P_verdadero|
ε_bit = Σ_bits (bit_i(error) · 2^i) / Σ_bits 2^i        # error normalizado por bit
risk  = (1/λ) · ln( mean( exp(λ · ε_bit) ) ),   λ = 20
reward = −10 · risk ,  recortado a [−100, 0]
```

- Reward = 0 → circuito perfecto.
- Reward ≈ −10 → error máximo en todos los casos.
- El `softmax`/log-sum-exp hace la señal más sensible a errores grandes (aversión al
  peor caso) en lugar de promediar errores.

### 6.7 Evaluación — modo incremental

Con `incremental=True`, el cursor llena la tabla **por columnas, de LSB a MSB**
(column-major). `_cursor_to_flat` mapea:

```
bit_pos = cursor // height        # columna lógica (0 = LSB)
row     = cursor % height
col     = grid_size − 1 − bit_pos # columna física
flat    = row · grid_size + col
```

Cada vez que se completa una columna (`cursor % height == 0`) se evalúa:

1. Se deduplican los términos de la columna (loop sobre `height`, tensores `(n, Proof)`
   — evita materializar `(n, Proof, n_actions)`).
2. `total = col_sum + carry_in`.
3. `sum_bit = total & 1`, `carry_out = total >> 1` (carry actualizado en `carry_in`).
4. El bit esperado es `(P_verdadero >> bit_pos) & 1`; `error_mean` = fracción de casos
   con el bit mal.
5. **Recompensa por columna** (dual):

   ```
   weight = 2^bit_pos
   norm   = 2^grid_size − 1
   col_reward =  +weight/norm     si error == 0   (bonus por columna correcta)
                 −(weight/norm)·error_mean  si error > 0
   ```

6. Al terminar todas las columnas, `_penalize_overflow`: si queda `carry_in > 0`
   (un multiplicador correcto no produce carry residual), se aplica
   `−10 · mean(carry > 0)`.

El reward incremental está en `[−10, 0]` para un circuito perfecto acumulado a 0.
Ventaja para MCTS: detecta mejoras parciales por columna sin esperar al final.

### 6.8 Logging por columna

`enable_column_logging()` (solo incremental) habilita buffers:

- `_col_error_log (n_envs, grid_size)` — error medio XOR por columna.
- `_carry_log (n_envs, grid_size+1)` — carry medio por columna.

Consultables con `get_column_stats()`; resetables con `reset_column_logs()`.

---

## 7. `BinaryMathEnvCUDAOptimized` — GEMM único con streaming por chunks

Resuelve el **OOM (out-of-memory)** de `BinaryMathEnvCUDA` para `Bits` grandes
(expandir `2^(2·Bits)` pares a GPU es inviable). Tres ideas combinadas:

1. **Caché de `action_vals`** — el valor de cada acción solo depende de los pares
   exhaustivos y del espacio de acciones, NO del entorno ni de `n_envs`. Se calcula
   una vez (`_build_eval_tables`) y se reutiliza; si no cabe en memoria
   (`ACTION_VALS_BUDGET_MB`, default 1024 MB), se recalcula por chunk
   (`_action_vals_for`).
2. **Evaluación con un único GEMM** en float32 — la rejilla se convierte en una matriz
   de presencia `(n_envs, n_actions)` con los factores de desplazamiento ya plegados
   (`_grid_presence`), y cada chunk de test cases se evalúa con `presence @ action_vals.T`
   vía cuBLAS: `O(n_envs·n_actions·chunk)`. TF32 se desactiva en el GEMM por exactitud
   (los productos son enteros y su cota realista ~4e6 para Bits=8 cabe en los 24 bits
   de mantisa float32).
3. **`commit_episodes` / `evaluate_grids`** — atajos para agentes cuya política solo
   depende del cursor: escriben un episodio completo (o una rejilla arbitraria) y lo
   evalúan de una sola vez, sin pagar CC iteraciones de Python con sus sincronizaciones
   CPU↔GPU. El chunking (`chunk_size` auto-ajustado por presupuesto de memoria) evita
   mantener tensores masivos.

Recompensa simplificada (sin riesgo exponencial, idéntica en valor):

```
avg_error = Σ_p |P_calc − P_true| / n_test_cases   (error normalizado por bit)
reward    = −10 · avg_error,  clamp a ≥ −100
```

```
Memoria: O(n_envs·chunk_size)  en vez de  O(n_envs · 2^(2·Bits))
Bits=8, n_envs=64     627 ms → 0.93 ms   (1875 MB → 122 MB)
Bits=8, n_envs=1024      OOM → 13.5 ms   (846 MB)
Bits=4, n_envs=4096    51.9 ms → 0.93 ms (1616 MB → 30 MB)
```

```python
BinaryMathEnvCUDAOptimized(Bits=8, Proof=4, height=8, n_envs=256,
                           device='cuda', chunk_size=None)
```

**Diferencias frente a `BinaryMathEnvCUDA`:**

| Característica | CUDA | CUDAOptimized |
|---|---|---|
| Test cases exhaustivos | tensores en GPU | generados por chunk |
| `action_vals` | recomputa por evaluación | cacheado (una vez, sin eje n_envs) |
| Evaluación | one-hot + einsum | un único GEMM (presence @ action_vals) |
| Modo incremental | sí | no |
| `no_repeat` / masking | sí | no |
| Memoria | `O(n_envs·2^(2·Bits))` | `O(n_envs·chunk_size)` |

---

## 8. Comparativa de las tres variantes

| Propiedad | `BinaryMathEnv` | `BinaryMathEnvCUDA` | `BinaryMathEnvCUDAOptimized` |
|---|---|---|---|
| **Interfaz** | Gym (`reset/step`) | Batched tensorial | Batched tensorial |
| **Evaluación** | iverilog + vvp | PyTorch GPU | PyTorch GPU (chunks) |
| **Casos de prueba** | `Proof` aleatorios | exhaustivos fijos | exhaustivos por chunk |
| **Determinista** | no (estocástico) | sí | sí |
| **Rango de recompensa** | `100 − 100·error` (hasta −50) | `[−100, 0]` | `[−100, 0]` |
| **Recompensa intermedia** | no (solo final) | sí si `incremental` | no |
| **Memoria** | mínima | `O(2^(2·Bits))` | `O(chunk_size)` |
| **Velocidad** | lenta (~100 ms/ep) | muy rápida | muy rápida |
| **Escala típica** | Bits ≤ 4 | Bits ≤ 4 | Bits ≤ 8 |
| **Dependencias extra** | iverilog/vvp | torch | torch |

**Formulario de recompensa resumido:**

```
CPU:            R = 100 − 100·E[ |P_sim − P| / (P + 1e-9) ]        (o −50 si falla)
CUDA:           R = −10·(1/λ)·ln(E[ e^{λ·ε_bit} ]),  λ=20,  clamp ≥ −100
CUDA incremental: R_col = (2^b / (2^g−1)) · ( +1 si ok, −error si mal )   + overflow −10·E[carry>0]
```

donde `b = bit_pos`, `g = grid_size`.

---

## 9. Reproducibilidad y dependencias

- **Semillas**: `BinaryMathEnv.reset(seed=...)` (API Gym) y, para fijar la secuencia de
  casos de prueba aleatorios de `closed()`, sembrar `np.random` global antes del
  entrenamiento. Las versiones CUDA no necesitan semilla (evaluación determinista).
- **Dependencias**:
  - Mínimas (CPU): `numpy`, `gymnasium`.
  - Render: `pygame`.
  - GPU: `torch` (+ CUDA si se dispone de GPU; las clases CUDA caen a CPU si no hay).
  - Evaluación Verilog: Icarus Verilog (`iverilog`, `vvp`).
- **Rutas**: las rutas de archivos (`multiplier.v`, `multipliermax.v`,
  `multiplier_8bit_tb.v`, `simv`, `testbench_template.v`) son relativas al paquete
  (`out/verilog/`). El directorio se crea automáticamente al generar Verilog.

---

## 10. Limitaciones y supuestos conocidos

1. **Observación no-Markoviana**: el estado no revela el contenido de la rejilla; los
   métodos tabulares aprenden una política heurística, no necesariamente óptima.
2. **Estocasticidad del CPU**: `Proof` casos aleatorios por episodio → recompensa con
   varianza. La misma tabla no siempre recibe la misma recompensa.
3. **Costo de evaluación**: cada episodio del CPU compila y simula Verilog (~100 ms), lo
   que limita el rendimiento de entrenamiento.
4. **Recompensa por complejidad**: en el CPU la penalidad de complejidad se calcula pero
   **no** se resta de la recompensa (la fórmula final solo usa el error funcional). Las
   versiones CUDA tampoco penalizan complejidad.
5. **Bits grandes**: el espacio de acciones (`2 + 4·Bits²`) y la longitud del episodio
   (`height·2·Bits`) crecen cuadráticamente; solo la variante CUDAOptimized escala de
   forma práctica.
6. **`maxi` y fases múltiples**: `maxi` no modifica la dinámica y el entorno opera en una
   única fase (`phase_names`), pese a estar diseñado para admitir más fases.
7. **Bit de signo / formato**: los test cases excluyen el 0 (`randint(1, 2^Bits)`) para
   evitar división por cero; el producto esperado se interpreta como entero sin signo.

---

## 11. Referencias cruzadas (archivo → concepto)

| Archivo | Conceptos |
|---|---|
| `envs/base.py` | `BinaryMathEnv`, espacios de acción/observación, `step`/`reset`, `closed`, `generate_verilog`, métricas de circuito, `clone`/`get_state`/`set_state` |
| `envs/pygame_env.py` | `BinaryMathEnvSecuencial`, render pygame, `clone`/`set_state` con recursos no serializables |
| `envs/cuda.py` | `BinaryMathEnvCUDA`, tensores de estado, test cases exhaustivos, modo incremental, action-masking, interfaz MCTS, logging |
| `envs/cuda_optimized.py` | `BinaryMathEnvCUDAOptimized`, streaming por chunks, reducción de memoria |
| `envs/__init__.py` | Re-exports y imports perezosos (PEP 562) |
| `verification/testbench_template.v` | Plantilla del testbench con placeholders `{regsI}`, `{regsO}`, `{Test}` |
| `out/verilog/multiplier.v` *(generado)* | Circuito sintetizado a partir de la tabla |
| `out/verilog/multipliermax.v` *(generado)* | Mejor circuito archivado (`min_error`) |
