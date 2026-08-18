PYTHON := .venv/bin/python
EPISODES := 10000000
N_ENVS := 512
EARLY_STOP := none

BITS := 2 4 6

.PHONY: all qlearning-cuda clean

all: qlearning-cuda

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

clean:
	rm -rf out/qlearning_cuda