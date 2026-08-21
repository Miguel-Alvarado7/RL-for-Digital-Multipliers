PYTHON := python3
EPISODES := 10000000
N_ENVS := 512
EARLY_STOP := none

BITS := 2 4 6

# Pasos de lookahead del retorno n-step SARSA. Solo lo usa train_sarsa_cuda.py;
# n_steps=1 es SARSA(0) clásico.
N_STEPS := 1 20 30

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

# Entrenamiento SARSA CUDA: producto cartesiano de 2, 4 y 6 bits (height = bits)
# por n_steps = 1, 20 y 30, con 512 entornos, 10M episodios y early stop
# desactivado. Son 9 corridas; cada una escribe en
# out/sarsa_cuda/bits<B>_nsteps<N>/.
#
# Para una corrida suelta, sobrescribe las listas desde la línea de comandos:
#   make sarsa-cuda BITS=4 N_STEPS=20
sarsa-cuda:
	@for n in $(N_STEPS); do \
		for b in $(BITS); do \
			echo "=== SARSA CUDA bits=$$b height=$$b n_steps=$$n ==="; \
			$(PYTHON) train_sarsa_cuda.py \
				--bits $$b \
				--height $$b \
				--n-envs $(N_ENVS) \
				--episodes $(EPISODES) \
				--n-steps $$n \
				--early-stop $(EARLY_STOP); \
		done; \
	done

clean:
	rm -rf out/qlearning_cuda out/montecarlo_cuda out/sarsa_cuda
