PYTHON := python3
EPISODES := 10000000
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

.PHONY: all qlearning-cuda montecarlo-cuda sarsa-cuda clean

all: qlearning-cuda montecarlo-cuda sarsa-cuda

# Entrenamiento Q-learning CUDA para 2, 4 y 6 bits con height = bits,
# 512 entornos, 10M episodios y early stop desactivado.
qlearning-cuda:
	@for b in $(BITS); do \
		echo "=== Q-learning CUDA bits=$$b height=$$b ==="; \
		$(PYTHON) train_qlearning_cuda.py \
			--bits $$b \
			--height $$b \
			--n-envs $(N_ENVS) \
			--episodes $(EPISODES) \
			--early-stop $(EARLY_STOP); \
	done

# Entrenamiento Monte Carlo CUDA para 2, 4 y 6 bits con height = bits,
# 512 entornos, 10M episodios y early stop desactivado.
montecarlo-cuda:
	@for b in $(BITS); do \
		echo "=== Monte Carlo CUDA bits=$$b height=$$b ==="; \
		$(PYTHON) train_mc_cuda.py \
			--bits $$b \
			--height $$b \
			--n-envs $(N_ENVS) \
			--episodes $(EPISODES) \
			--early-stop $(EARLY_STOP); \
	done

# Entrenamiento SARSA CUDA: producto cartesiano de 2, 4 y 6 bits por
# n_steps = 1, 20 y 30, con 512 entornos, 10M episodios y early stop
# desactivado. height = bits salvo a 4 bits, donde se usa SARSA_HEIGHT_4 (=3).
# Son 9 corridas; cada una escribe en out/sarsa_cuda/bits<B>_nsteps<N>/.
#
# Para una corrida suelta, sobrescribe las listas desde la línea de comandos:
#   make sarsa-cuda BITS=4 N_STEPS=20
sarsa-cuda:
	@for n in $(N_STEPS); do \
		for b in $(BITS); do \
			case $$b in \
				4) h=$(SARSA_HEIGHT_4) ;; \
				*) h=$$b ;; \
			esac; \
			echo "=== SARSA CUDA bits=$$b height=$$h n_steps=$$n ==="; \
			$(PYTHON) train_sarsa_cuda.py \
				--bits $$b \
				--height $$h \
				--n-envs $(N_ENVS) \
				--episodes $(EPISODES) \
				--n-steps $$n \
				--early-stop $(EARLY_STOP); \
		done; \
	done

clean:
	rm -rf out/qlearning_cuda out/montecarlo_cuda out/sarsa_cuda
