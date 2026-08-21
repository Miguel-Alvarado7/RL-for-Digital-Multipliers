"""
Optimized CUDA Environment with Memory-Efficient Streaming.

Soluciona el problema de OOM en BinaryMathEnvCUDA eliminando la expansión
masiva de pares (A,B) a GPU. En su lugar:

1. Cachea la tabla de valores de acción (constante: depende solo de los pares
   exhaustivos y del espacio de acciones, NO del entorno ni de n_envs)
2. Evalúa cada rejilla con un único GEMM en float32, con los factores de
   desplazamiento ya plegados en la matriz de presencia
3. Procesa los casos de prueba en chunks para acotar la memoria
4. Mantiene velocidad GPU: cálculos vectorizados, no por unidad

Medido en una RTX 4000 Ada, batch de entrenamiento completo (n_envs episodios
+ update de Q), contra la versión que expandía los pares (A,B) a n_envs:

    Bits=8, n_envs=64     627 ms  ->  0.93 ms   (1875 MB -> 122 MB)
    Bits=8, n_envs=1024      OOM  ->  13.5 ms   (846 MB)
    Bits=4, n_envs=4096    51.9 ms -> 0.93 ms   (1616 MB ->  30 MB)

Uso:
    cuda_env = BinaryMathEnvCUDAOptimized(Bits=8, height=8, n_envs=256)
    rewards = cuda_env.rollout_from_state(state, n_rollouts=256)
"""

import torch
import numpy as np
from typing import Tuple, Dict, List
import random


class BinaryMathEnvCUDAOptimized:
    """
    Versión memory-optimized de BinaryMathEnvCUDA con streaming de evaluaciones.

    Diferencia clave: En lugar de cargar TODOS los pares (A,B) a GPU,
    genera y procesa chunks bajo demanda. Esto permite Bits=8 sin OOM.
    """

    #: Presupuesto para el tensor de productos (n_envs, chunk) al auto-elegir
    #: chunk_size.
    PRODUCT_BUDGET_MB = 256
    #: Presupuesto para cachear action_vals (n_test_cases, n_actions) float32.
    #: Si la tabla no cabe, se recalcula por chunk (sin expandir a n_envs).
    ACTION_VALS_BUDGET_MB = 1024

    def __init__(
        self,
        Bits: int = 8,
        Proof: int = 4,
        height: int = 8,
        n_envs: int = 256,
        device: str = 'cuda',
        chunk_size: int = None,
        error_mode: str = 'wrap',
        area_lambda: float = 0.0,
    ):
        """
        Args:
            Bits:       Número de bits del multiplicador.
            Proof:      Parámetro heredado (se reemplaza con n_test_cases).
            height:     Filas de la tabla de productos parciales.
            n_envs:     Número de entornos en paralelo.
            device:     'cuda' o 'cpu'. Cae a CPU si CUDA no disponible.
            chunk_size: Pares (A,B) por chunk de evaluación. None => auto, se
                        elige para que el tensor de productos (n_envs, chunk)
                        quepa en ~PRODUCT_BUDGET_MB. Un entero lo fuerza.
            error_mode: 'wrap' (default, comportamiento histórico) envuelve el
                        error módulo 2^max_bits, así que un circuito que se
                        pasa por mucho puntúa como si se pasara por poco;
                        'saturate' lo satura en el máximo (deja de premiar el
                        desbordamiento, pero aplana la zona mala);
                        'linear' usa el error real sin envolver ni aplanar.
            area_lambda: Penalización por término usado, restada del reward.
                        0 = desactivada (comportamiento histórico).
        """
        if error_mode not in ('wrap', 'saturate', 'linear'):
            raise ValueError(
                f"error_mode debe ser 'wrap', 'saturate' o 'linear', no {error_mode!r}")
        self.error_mode = error_mode
        self.area_lambda = float(area_lambda)
        self.Bits = Bits
        self.height = height
        self.n_envs = n_envs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.CC = height * 2 * Bits
        self.grid_size = 2 * Bits

        # Construir espacio de acciones
        self._build_action_space()
        # Pre-calcular tablas de decodificación
        self._build_decode_tables()

        # Estado del entorno: igual que antes pero sin test cases masivos
        self.suma_grid = torch.full(
            (n_envs, self.CC), -1, dtype=torch.int16, device=self.device
        )
        self.cursor_pos = torch.zeros(n_envs, dtype=torch.int32, device=self.device)
        self.done = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self.rewards = torch.zeros(n_envs, dtype=torch.float32, device=self.device)

        # Información de pares exhaustivos (solo metadatos, NO se expanden a GPU)
        max_val = 2 ** self.Bits
        self.n_test_cases = max_val * max_val
        self.max_val = max_val
        self.max_product = (max_val - 1) ** 2

        # Actualizar Proof para reflejar exhaustividad
        self.Proof = self.n_test_cases

        # Chunking sobre casos de prueba. El tensor dominante de la evaluación
        # es products (n_envs, chunk) float32; se elige chunk para acotarlo.
        if chunk_size is None:
            budget = int(self.PRODUCT_BUDGET_MB * 2 ** 20 / (4 * max(n_envs, 1)))
            chunk_size = max(1024, min(self.n_test_cases, budget))
        self.chunk_size = int(chunk_size)

        # Constantes de evaluación: dependen solo de (A,B) y del espacio de
        # acciones, que son fijos durante toda la vida del entorno.
        self._build_eval_tables()

        print(
            f"[CUDAOptimized] Bits={Bits}, n_test_cases={self.n_test_cases}, "
            f"chunk_size={self.chunk_size}, cache_action_vals="
            f"{'si' if self._action_vals is not None else 'no'}, "
            f"device={self.device}"
        )

    # =========================================================================
    # Construcción interna
    # =========================================================================

    def _build_action_space(self):
        """Mismo orden que BinaryMathEnv."""
        self.possible_actions = ['0', '1']
        for i in range(self.Bits):
            for j in range(self.Bits):
                self.possible_actions.append(f'(A[{i}]&B[{j}])')
                self.possible_actions.append(f'(~A[{i}]&B[{j}])')
                self.possible_actions.append(f'(A[{i}]&~B[{j}])')
                self.possible_actions.append(f'(~A[{i}]&~B[{j}])')
        self.n_actions = len(self.possible_actions)
        self._action_to_idx = {a: i for i, a in enumerate(self.possible_actions)}

    def _build_decode_tables(self):
        """Pre-calcula decodificación de acciones en GPU."""
        n_pp = self.n_actions - 2
        i_bits = torch.empty(n_pp, dtype=torch.int64)
        j_bits = torch.empty(n_pp, dtype=torch.int64)
        neg_a = torch.empty(n_pp, dtype=torch.bool)
        neg_b = torch.empty(n_pp, dtype=torch.bool)

        k = 0
        for i in range(self.Bits):
            for j in range(self.Bits):
                i_bits[k] = i
                j_bits[k] = j
                neg_a[k] = False
                neg_b[k] = False
                k += 1
                i_bits[k] = i
                j_bits[k] = j
                neg_a[k] = True
                neg_b[k] = False
                k += 1
                i_bits[k] = i
                j_bits[k] = j
                neg_a[k] = False
                neg_b[k] = True
                k += 1
                i_bits[k] = i
                j_bits[k] = j
                neg_a[k] = True
                neg_b[k] = True
                k += 1

        self.action_i_bits = i_bits.to(self.device)
        self.action_j_bits = j_bits.to(self.device)
        self.action_neg_a = neg_a.to(self.device)
        self.action_neg_b = neg_b.to(self.device)

        shifts = torch.tensor(
            [self.grid_size - col - 1 for col in range(self.grid_size)],
            dtype=torch.float64,
        )
        self.shift_factors = (2.0 ** shifts).to(self.device)

    def _build_eval_tables(self):
        """Pre-calcula lo que NO depende de la rejilla ni de n_envs.

        `action_vals[p, a]` (valor del bit que produce la acción `a` para el
        par de test `p`) y `true_P[p]` (= A·B) dependen solo de los pares
        exhaustivos, que son fijos. Calcularlos una vez evita repetirlos en
        cada evaluación, y evita la expansión a (n_envs, chunk, n_actions) que
        era la causa real de los OOM: la tabla es la MISMA para todo n_envs.

        Todo en float32: los productos son enteros y su cota realista
        (~2^22 para Bits=8, ver _compute_products_chunked) cabe exacta en los
        24 bits de mantisa. float64 en GPUs de consumo va a 1/64 del ritmo.
        """
        # Error normalizado: la descomposición en bits del entorno original,
        #   sum_k bit_k(e)·2^k / sum_k 2^k,
        # es idénticamente  (e mod 2^max_bits) / (2^max_bits - 1).
        self._max_bits = int(np.ceil(np.log2(self.max_product + 1)))
        self._err_mod = float(2 ** self._max_bits)
        self._err_denom = float(2 ** self._max_bits - 1)

        self._shift_f32 = self.shift_factors.float()

        idx = torch.arange(self.n_test_cases, device=self.device)
        self._true_P = ((idx // self.max_val) * (idx % self.max_val)).float()

        vals_mb = self.n_test_cases * self.n_actions * 4 / 2 ** 20
        if vals_mb <= self.ACTION_VALS_BUDGET_MB:
            self._action_vals = self._compute_action_vals(0, self.n_test_cases)
        else:
            self._action_vals = None  # se recalcula por chunk

    def _compute_action_vals(self, lo: int, hi: int) -> torch.Tensor:
        """Tabla (hi-lo, n_actions) float32 con el valor de cada acción.

        Sin eje n_envs: el valor de una acción depende del par (A,B), no del
        entorno que la eligió.
        """
        idx = torch.arange(lo, hi, device=self.device)
        a_vals, b_vals = idx // self.max_val, idx % self.max_val

        A_bits = ((a_vals.unsqueeze(-1) >> self.action_i_bits) & 1).float()
        B_bits = ((b_vals.unsqueeze(-1) >> self.action_j_bits) & 1).float()
        A_eff = torch.where(self.action_neg_a, 1.0 - A_bits, A_bits)
        B_eff = torch.where(self.action_neg_b, 1.0 - B_bits, B_bits)

        vals = torch.zeros(hi - lo, self.n_actions, dtype=torch.float32,
                           device=self.device)
        vals[:, 1] = 1.0            # acción '1'; la acción '0' queda en 0
        vals[:, 2:] = A_eff * B_eff
        return vals.contiguous()

    def _action_vals_for(self, lo: int, hi: int) -> torch.Tensor:
        """Slice cacheado de action_vals, o recálculo si no cabía en memoria."""
        if self._action_vals is not None:
            return self._action_vals[lo:hi]
        return self._compute_action_vals(lo, hi)

    # =========================================================================
    # API principal (compatible con BinaryMathEnv)
    # =========================================================================

    def reset(self, env_indices=None):
        """Reinicia entornos."""
        if env_indices is None:
            self.suma_grid.fill_(-1)
            self.cursor_pos.fill_(0)
            self.done.fill_(False)
            self.rewards.fill_(0.0)
        else:
            idx = torch.as_tensor(env_indices, device=self.device)
            self.suma_grid[idx] = -1
            self.cursor_pos[idx] = 0
            self.done[idx] = False
            self.rewards[idx] = 0.0

    def step(self, actions):
        """Ejecuta un paso para todos los entornos."""
        active = ~self.done
        writable = active & (self.cursor_pos < self.CC)

        if writable.any():
            idx = writable.nonzero(as_tuple=True)[0]
            cols = self.cursor_pos[idx]
            self.suma_grid[idx, cols] = actions[idx].to(torch.int16)
            self.cursor_pos[idx] += 1

        just_done = writable & (self.cursor_pos >= self.CC)
        step_rewards = torch.zeros(self.n_envs, device=self.device)
        if just_done.any():
            r = self._evaluate_batch(just_done)
            self.rewards[just_done] = r
            step_rewards[just_done] = r
            self.done[just_done] = True

        return step_rewards, self.done.clone()

    def commit_episodes(self, actions: torch.Tensor) -> torch.Tensor:
        """Escribe episodios COMPLETOS y los evalúa de una sola vez.

        Atajo para agentes cuya política no depende del contenido de la
        rejilla (solo del cursor): en ese caso las CC acciones de un episodio
        se pueden decidir por adelantado, y no hay razón para pagar CC
        iteraciones de Python con sus sincronizaciones CPU<->GPU. Deja el
        entorno en el mismo estado que CC llamadas a step().

        Args:
            actions: (n_envs, CC) int — acción por celda, en orden de cursor.

        Returns:
            (n_envs,) float32 — reward terminal de cada entorno.
        """
        if actions.shape != (self.n_envs, self.CC):
            raise ValueError(
                f"actions debe ser ({self.n_envs}, {self.CC}), "
                f"recibido {tuple(actions.shape)}"
            )
        self.suma_grid.copy_(actions.to(torch.int16))
        self.cursor_pos.fill_(self.CC)
        self.done.fill_(True)
        self.rewards = self._compute_products_chunked(self.suma_grid.long())
        return self.rewards

    def evaluate_grids(self, grids: torch.Tensor) -> torch.Tensor:
        """Evalúa rejillas arbitrarias sin tocar el estado del entorno.

        Args:
            grids: (n, CC) int — índices de acción, -1 = celda vacía.

        Returns:
            (n,) float32 — reward de cada rejilla.
        """
        return self._compute_products_chunked(grids.long())

    # =========================================================================
    # Evaluación eficiente por chunks (CORE OPTIMIZATION)
    # =========================================================================

    def _evaluate_all_actions(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula valor bit de cada acción para test cases dados.

        Conservado por compatibilidad de API. NO se usa en el camino caliente:
        el valor de una acción no depende del entorno, así que replicarlo por
        n_envs (como hacía la versión anterior) es trabajo redundante puro.
        Prefiere `_action_vals_for(lo, hi)`.

        Args:
            A: (n_chunks, chunk_size) o (chunk_size,)
            B: (n_chunks, chunk_size) o (chunk_size,)

        Returns:
            (n_chunks, chunk_size, n_actions) float32
        """
        A = A.to(self.device)
        B = B.to(self.device)
        if A.dim() == 1:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)

        n, m = A.shape[0], A.shape[1]
        vals = torch.zeros(n, m, self.n_actions, dtype=torch.float32, device=self.device)
        vals[:, :, 1] = 1.0

        A_bits = ((A.unsqueeze(-1) >> self.action_i_bits) & 1).float()
        B_bits = ((B.unsqueeze(-1) >> self.action_j_bits) & 1).float()
        A_eff = torch.where(self.action_neg_a, 1.0 - A_bits, A_bits)
        B_eff = torch.where(self.action_neg_b, 1.0 - B_bits, B_bits)

        vals[:, :, 2:] = A_eff * B_eff
        return vals

    def _grid_presence(self, grids: torch.Tensor) -> torch.Tensor:
        """presence[e, a] = peso posicional total de la acción `a` en la rejilla e.

        La rejilla se lee como (height, grid_size); una acción cuenta UNA vez
        por columna aunque se repita en varias filas (deduplicación sobre
        height), y cada columna aporta su factor 2^(grid_size-c-1).

        Plegar los shifts aquí -- en vez de después de contraer por acciones --
        elimina el eje de columnas de la contracción pesada: la evaluación pasa
        de O(n_envs·grid_size·n_actions·chunk) a O(n_envs·n_actions·chunk).

        Args:
            grids: (n_envs, CC) int — índices de acción, -1 = celda vacía.

        Returns:
            presence: (n_envs, n_actions) float32 — pesos plegados.
            terms:    (n_envs,) float32 — nº de términos distintos != '0'
                      sumados sobre columnas. Proxy de área del circuito.
        """
        n_envs = grids.shape[0]
        g = grids.view(n_envs, self.height, self.grid_size).long()
        # Las celdas vacías (-1) van a un slot basura que luego se descarta.
        g = torch.where(g >= 0, g, torch.full_like(g, self.n_actions))

        presence = torch.zeros(n_envs, self.grid_size, self.n_actions + 1,
                               dtype=torch.float32, device=self.device)
        # scatter de 1s: escribir el mismo 1 varias veces == max sobre height.
        presence.scatter_(2, g.transpose(1, 2), 1.0)
        presence = presence[:, :, :self.n_actions]

        # Área: términos distintos por columna, sin contar la acción '0'
        # (índice 0), que no aporta nada al circuito.
        terms = presence[:, :, 1:].sum(dim=(1, 2))

        return torch.einsum('eca,c->ea', presence, self._shift_f32), terms

    def _compute_products_chunked(
        self, grids: torch.Tensor
    ) -> torch.Tensor:
        """
        Evalúa producto (A*B) contra los pares exhaustivos, en chunks.

        Con los shifts ya plegados en `presence`, el producto de cada entorno
        para cada par de test es un único GEMM (n_envs, n_actions) x
        (n_actions, chunk), que va a cuBLAS.

        Exactitud en float32: los productos son enteros y su cota es
        n_actions·(2^grid_size - 1) en el peor caso teórico, pero A[i]&B[j] y
        ~A[i]&B[j] son mutuamente excluyentes, así que a lo sumo ~1/4 de los
        productos parciales valen 1 a la vez: ~65·65535 ≈ 4.3e6 para Bits=8,
        holgadamente dentro de los 2^24 exactos de float32. Por eso se
        desactiva TF32 (mantisa de 10 bits) en el GEMM.

        Args:
            grids: (n_envs, CC) - índices de acciones

        Returns:
            (n_envs,) float32 - rewards normalizados
        """
        n_envs = grids.shape[0]
        presence, terms = self._grid_presence(grids)   # (n_envs, n_actions)

        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            error_sums = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
            for lo in range(0, self.n_test_cases, self.chunk_size):
                hi = min(lo + self.chunk_size, self.n_test_cases)

                action_vals = self._action_vals_for(lo, hi)     # (chunk, n_actions)
                products = presence @ action_vals.T             # (n_envs, chunk)

                error = (products - self._true_P[lo:hi].unsqueeze(0)).abs()
                error.floor_()
                if self.error_mode == 'linear':
                    # Error real, sin envolver ni aplanar: monotono en todo el
                    # rango, asi que nunca premia pasarse y conserva gradiente
                    # tambien en la zona mala (donde 'saturate' es una meseta).
                    normalized = error
                elif self.error_mode == 'saturate':
                    # Un error mayor que el maximo representable es, como
                    # minimo, tan malo como el maximo: se satura en vez de
                    # envolverse.
                    normalized = error.clamp(max=self._err_denom)
                else:
                    # 'wrap': equivalente exacto de la suma bit-ponderada
                    # del entorno original (error mod 2^max_bits).
                    normalized = torch.remainder(error, self._err_mod)
                error_sums += normalized.sum(dim=1) / self._err_denom
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

        avg_error = error_sums / self.n_test_cases
        # El original calculaba log(exp(l*avg_error))/l, que es la identidad
        # (y desborda a inf si l*avg_error > 709). Se aplica directamente.
        reward = (-10.0 * avg_error).clamp(min=-100.0)
        if self.area_lambda:
            # Penalizacion de area: sin esto nada empuja al agente a usar
            # menos terminos, y 'height' acaba haciendo de proxy de area.
            reward = reward - self.area_lambda * terms
        return reward

    def true_metrics(self, grids: torch.Tensor):
        """Métricas honestas de un circuito, independientes del reward.

        No envuelve el error ni aplica penalización de área, así que sirve
        como patrón neutral para comparar funciones de reward distintas.

        Args:
            grids: (n, CC) int — índices de acción.

        Returns:
            mae:   (n,) float32 — error absoluto medio sobre todos los pares.
            terms: (n,) float32 — términos distintos != '0' (proxy de área).
            exact: (n,) float32 — fracción de pares calculados exactamente.
            mred:  (n,) float32 — error relativo medio (|err|/A·B), promediado
                   solo sobre los pares con A·B > 0 (el estándar en
                   computación aproximada; A·B = 0 lo haría indefinido).
        """
        presence, terms = self._grid_presence(grids.long())
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            n = grids.shape[0]
            err_sum = torch.zeros(n, device=self.device)
            exact = torch.zeros(n, device=self.device)
            rel_sum = torch.zeros(n, device=self.device)
            n_nz = 0
            for lo in range(0, self.n_test_cases, self.chunk_size):
                hi = min(lo + self.chunk_size, self.n_test_cases)
                T = self._true_P[lo:hi]
                P = presence @ self._action_vals_for(lo, hi).T
                e = (P - T.unsqueeze(0)).abs()
                err_sum += e.sum(dim=1)
                exact += (e == 0).float().sum(dim=1)
                nz = T > 0
                rel_sum += (e[:, nz] / T[nz]).sum(dim=1)
                n_nz += int(nz.sum())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        return (err_sum / self.n_test_cases, terms,
                exact / self.n_test_cases, rel_sum / max(n_nz, 1))

    def _evaluate_batch(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Evalúa entornos enmascarados usando chunking."""
        idx = env_mask.nonzero(as_tuple=True)[0]
        grids = self.suma_grid[idx].long()
        rewards = self._compute_products_chunked(grids)
        return rewards.float()  # Asegurar que devuelve float32

    # =========================================================================
    # Interfaz MCTS
    # =========================================================================

    def rollout_from_state(
        self,
        state_dict: Dict,
        n_rollouts: int = None,
        rollout_depth: int = None,
    ) -> torch.Tensor:
        """
        Ejecuta N rollouts aleatorios paralelos desde estado MCTS.

        Los rollouts se evalúan con TODOS los pares (A,B) exhaustivos,
        pero procesados en chunks para evitar OOM.
        """
        if n_rollouts is None:
            n_rollouts = self.n_envs
        n_rollouts = min(n_rollouts, self.n_envs)

        if rollout_depth is None:
            rollout_depth = self.CC

        env_idx = list(range(n_rollouts))
        self.reset(env_idx)
        self._load_state(state_dict, env_idx)

        # Rollouts aleatorios
        for _ in range(rollout_depth):
            active = ~self.done[:n_rollouts]
            if not active.any():
                break
            actions = torch.randint(
                0, self.n_actions, (n_rollouts,), device=self.device
            )
            self._step_range(actions, n_rollouts)

        return self.rewards[:n_rollouts].clone()

    def _step_range(self, actions: torch.Tensor, n: int):
        """step() restringido a los primeros n entornos."""
        active = ~self.done[:n]
        writable = active & (self.cursor_pos[:n] < self.CC)

        if writable.any():
            idx = writable.nonzero(as_tuple=True)[0]
            cols = self.cursor_pos[idx]
            self.suma_grid[idx, cols] = actions[idx].to(torch.int16)
            self.cursor_pos[idx] += 1

        just_done = writable & (self.cursor_pos[:n] >= self.CC)
        if just_done.any():
            full_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            full_mask[:n] = just_done
            r = self._evaluate_batch(full_mask)
            self.rewards[full_mask] = r
            self.done[full_mask] = True

    # =========================================================================
    # Interoperabilidad con BinaryMathEnv
    # =========================================================================

    def _load_state(self, state_dict: Dict, env_indices: List[int]):
        """Carga estado de BinaryMathEnv."""
        grid_strs = state_dict['suma_grid']
        cursor_val = state_dict['cursor_position']

        grid_indices = [self._action_to_idx.get(s, -1) for s in grid_strs]
        grid_tensor = torch.tensor(grid_indices, dtype=torch.int16, device=self.device)

        idx = torch.as_tensor(env_indices, device=self.device)
        self.suma_grid[idx] = grid_tensor.unsqueeze(0)
        self.cursor_pos[idx] = cursor_val
        self.done[idx] = False
        self.rewards[idx] = 0.0

    def grid_to_state(self, grid_1d) -> Dict:
        """Convierte una rejilla 1D (índices de acción) en un dict de estado.

        Útil para serializar una rejilla sin depender del estado actual del
        entorno (p. ej. una rejilla ya sobrescrita por una corrida greedy).
        """
        grid = grid_1d.cpu().tolist() if torch.is_tensor(grid_1d) else list(grid_1d)
        grid_strs = [
            self.possible_actions[idx] if idx >= 0 else ' '
            for idx in grid
        ]
        return {
            'suma_grid': grid_strs,
            'cursor_position': int(self.CC),
        }

    def get_single_state(self, env_idx: int) -> Dict:
        """Devuelve estado de un entorno como dict."""
        return self.grid_to_state(self.suma_grid[env_idx])

    def clone_env(self, src_idx: int, dst_idx):
        """Clona entorno dentro del batch."""
        self.suma_grid[dst_idx] = self.suma_grid[src_idx]
        self.cursor_pos[dst_idx] = self.cursor_pos[src_idx]
        self.done[dst_idx] = self.done[src_idx]
        self.rewards[dst_idx] = self.rewards[src_idx]

    # =========================================================================
    # Estadísticas
    # =========================================================================

    @property
    def active_count(self) -> int:
        """Número de entornos en ejecución."""
        return int((~self.done).sum().item())

    @property
    def completed_count(self) -> int:
        """Número de entornos completados."""
        return int(self.done.sum().item())

    def summary(self):
        """Imprime estadísticas."""
        done_rewards = self.rewards[self.done]
        print(f"Device:     {self.device}")
        print(
            f"Entornos:   {self.n_envs}  |  activos: {self.active_count}  |  "
            f"completados: {self.completed_count}"
        )
        print(f"Test cases: {self.n_test_cases} (exhaustivo)")
        print(f"Chunk size: {self.chunk_size}")
        if done_rewards.numel() > 0:
            print(
                f"Reward      min={done_rewards.min():.2f}  "
                f"mean={done_rewards.mean():.2f}  max={done_rewards.max():.2f}"
            )
        else:
            print("Reward      sin datos aún")
