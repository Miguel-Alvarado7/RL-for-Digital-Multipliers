PYTHON := python3
EPISODES := 50000000
N_ENVS := 512
EARLY_STOP := none

BITS := 2 4 6

# Pasos de lookahead del retorno n-step SARSA. Solo lo usa train_sarsa_cuda.py;
# n_steps=1 es SARSA(0) clásico.
N_STEPS := 1 20 30

# Altura de la tabla para SARSA. Por defecto height = bits, salvo a 4 bits,
# donde height=3 mide mejor como aproximador: con SARSA n_steps=20, 1M
# episodios y 3 semillas, height=3 iguala el error relativo de height=4
# (16,3% vs 14,4%, dentro del solape entre semillas) y su peor error absoluto
# (13 en ambos) usando 3,3 terminos menos de area. height=2 queda claramente
# por detras (27,5% de error relativo, peor caso 45).
SARSA_HEIGHT_4 := 3

# Verbosidad. Con 97k batches por corrida el monitoreo greedy imprimia ~9.800
# lineas y el progreso ~980; --quiet elimina las primeras (y su computo, que no
# sirve para nada mas) y --log-every espacia las segundas. El early stop no se
# ve afectado: su corrida greedy de confirmacion se sigue imprimiendo.
#
# Para recuperar la salida completa:  make all QUIET= LOG_EVERY=100
QUIET := --quiet
LOG_EVERY := 1000

.PHONY: all qlearning-cuda montecarlo-cuda sarsa-cuda clean

# `all` se invoca a si mismo de forma recursiva en vez de declarar los targets
# como prerequisitos: make ejecuta los prerequisitos ANTES de la receta, asi
# que con `all: qlearning-cuda ...` no habria forma de marcar el instante de
# inicio.
#
# OJO: si una corrida falla, el bucle `for` de cada target continua y la receta
# termina en el printf, asi que make sale con 0 igual. El cronometro es la
# pista: una corrida que falla tarda segundos en vez de minutos.
all:
	@t0=$$(date +%s); \
	echo "### inicio: $$(date '+%Y-%m-%d %H:%M:%S') ###"; \
	$(MAKE) --no-print-directory qlearning-cuda montecarlo-cuda sarsa-cuda; \
	rc=$$?; \
	d=$$(( $$(date +%s) - t0 )); \
	echo ""; \
	echo "======================================================="; \
	printf '  TIEMPO TOTAL (make all): %02d:%02d:%02d  (%s s)\n' \
		$$((d/3600)) $$((d%3600/60)) $$((d%60)) "$$d"; \
	echo "  fin: $$(date '+%Y-%m-%d %H:%M:%S')"; \
	echo "======================================================="; \
	exit $$rc

# Entrenamiento Q-learning CUDA para los BITS indicados con height = bits,
# N_ENVS entornos, EPISODES episodios y early stop desactivado.
qlearning-cuda:
	@t0=$$(date +%s); \
	for b in $(BITS); do \
		echo "=== Q-learning CUDA bits=$$b height=$$b ==="; \
		r0=$$(date +%s); \
		$(PYTHON) train_qlearning_cuda.py \
			--bits $$b \
			--height $$b \
			--n-envs $(N_ENVS) \
			--episodes $(EPISODES) \
			--early-stop $(EARLY_STOP) \
			$(QUIET) --log-every $(LOG_EVERY); \
		d=$$(( $$(date +%s) - r0 )); \
		printf '    [tiempo] Q-learning bits=%s: %02d:%02d:%02d\n' \
			"$$b" $$((d/3600)) $$((d%3600/60)) $$((d%60)); \
	done; \
	d=$$(( $$(date +%s) - t0 )); \
	printf '=== [TIEMPO] qlearning-cuda: %02d:%02d:%02d ===\n' \
		$$((d/3600)) $$((d%3600/60)) $$((d%60))

# Entrenamiento Monte Carlo CUDA para los BITS indicados con height = bits,
# N_ENVS entornos, EPISODES episodios y early stop desactivado.
montecarlo-cuda:
	@t0=$$(date +%s); \
	for b in $(BITS); do \
		echo "=== Monte Carlo CUDA bits=$$b height=$$b ==="; \
		r0=$$(date +%s); \
		$(PYTHON) train_mc_cuda.py \
			--bits $$b \
			--height $$b \
			--n-envs $(N_ENVS) \
			--episodes $(EPISODES) \
			--early-stop $(EARLY_STOP) \
			$(QUIET) --log-every $(LOG_EVERY); \
		d=$$(( $$(date +%s) - r0 )); \
		printf '    [tiempo] Monte Carlo bits=%s: %02d:%02d:%02d\n' \
			"$$b" $$((d/3600)) $$((d%3600/60)) $$((d%60)); \
	done; \
	d=$$(( $$(date +%s) - t0 )); \
	printf '=== [TIEMPO] montecarlo-cuda: %02d:%02d:%02d ===\n' \
		$$((d/3600)) $$((d%3600/60)) $$((d%60))

# Entrenamiento SARSA CUDA: producto cartesiano de BITS por N_STEPS, con
# N_ENVS entornos, EPISODES episodios y early stop desactivado.
# height = bits salvo a 4 bits, donde se usa SARSA_HEIGHT_4 (=3).
# Son 9 corridas; cada una escribe en out/sarsa_cuda/bits<B>_nsteps<N>/.
#
# Para una corrida suelta, sobrescribe las listas desde la línea de comandos:
#   make sarsa-cuda BITS=4 N_STEPS=20
sarsa-cuda:
	@t0=$$(date +%s); \
	for n in $(N_STEPS); do \
		for b in $(BITS); do \
			case $$b in \
				4) h=$(SARSA_HEIGHT_4) ;; \
				*) h=$$b ;; \
			esac; \
			echo "=== SARSA CUDA bits=$$b height=$$h n_steps=$$n ==="; \
			r0=$$(date +%s); \
			$(PYTHON) train_sarsa_cuda.py \
				--bits $$b \
				--height $$h \
				--n-envs $(N_ENVS) \
				--episodes $(EPISODES) \
				--n-steps $$n \
				--early-stop $(EARLY_STOP) \
			$(QUIET) --log-every $(LOG_EVERY); \
			d=$$(( $$(date +%s) - r0 )); \
			printf '    [tiempo] SARSA bits=%s n_steps=%s: %02d:%02d:%02d\n' \
				"$$b" "$$n" $$((d/3600)) $$((d%3600/60)) $$((d%60)); \
		done; \
	done; \
	d=$$(( $$(date +%s) - t0 )); \
	printf '=== [TIEMPO] sarsa-cuda: %02d:%02d:%02d ===\n' \
		$$((d/3600)) $$((d%3600/60)) $$((d%60))

clean:
	rm -rf out/qlearning_cuda out/montecarlo_cuda out/sarsa_cuda
