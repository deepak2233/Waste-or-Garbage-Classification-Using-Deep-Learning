# Common tasks. `make help` lists them.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
DATA   ?= data/raw
CONFIG ?= configs/base.yaml
RUN    ?=

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev lint format typecheck test test-all test-fast \
        synthetic scan train evaluate smoke clean clean-runs docker

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: venv ## Install the package
	$(BIN)/pip install -e .

install-dev: venv ## Install with dev and serving extras
	$(BIN)/pip install -e ".[all]"

lint: ## Check style
	$(BIN)/ruff check src/ tests/ scripts/
	$(BIN)/ruff format --check src/ tests/ scripts/

format: ## Apply formatting and autofixes
	$(BIN)/ruff check --fix src/ tests/ scripts/
	$(BIN)/ruff format src/ tests/ scripts/

test: ## Run the fast tests
	$(BIN)/pytest tests/ -q

test-all: ## Run every test, including training end to end
	$(BIN)/pytest tests/ -q -m "slow or not slow"

test-fast: ## Run only the tests that do not need TensorFlow
	$(BIN)/pytest tests/ -q -m "not needs_tf and not slow"

synthetic: ## Generate a synthetic dataset under data/synthetic
	$(BIN)/python scripts/make_synthetic_data.py --out data/synthetic --per-class 40

scan: ## Report the dataset split
	$(BIN)/wasteclf scan --data-root $(DATA)

train: ## Train with CONFIG (default configs/base.yaml)
	$(BIN)/wasteclf train -c $(CONFIG) --data-root $(DATA)

evaluate: ## Evaluate RUN=runs/<name> on the test split
	@test -n "$(RUN)" || (echo "usage: make evaluate RUN=runs/<name>" && exit 1)
	$(BIN)/wasteclf evaluate --run $(RUN) --split test

smoke: synthetic ## End-to-end check on synthetic data
	$(BIN)/wasteclf train -c configs/smoke.yaml

data: ## Download the 12-class dataset (needs Kaggle credentials)
	$(BIN)/python scripts/fetch_data.py garbage12 --out $(DATA)

data-trashnet: ## Download TrashNet (public, no credentials)
	$(BIN)/python scripts/fetch_data.py trashnet --out data/trashnet

scene: ## Analyse a scene: make scene RUN=runs/<name> IMAGE=dump.jpg
	@test -n "$(RUN)" -a -n "$(IMAGE)" || (echo "usage: make scene RUN=runs/<name> IMAGE=dump.jpg" && exit 1)
	$(BIN)/wasteclf scene --run $(RUN) $(IMAGE) --overlay scenes/

docker: ## Build the container image
	docker build -t wasteclf:latest .

clean: ## Remove build and cache artefacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .coverage htmlcov/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-runs: ## Delete every training run (irreversible)
	rm -rf runs/
