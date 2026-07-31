PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.PHONY: setup data all ingest features models economics monitoring report window-scan components sweep brief hi-small test lint clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -r requirements.txt
	$(BIN)/pip freeze > requirements.lock
	@echo "--- requirements.lock ---"
	@cat requirements.lock
	@$(PY) -c "import xgboost, sklearn, pandas, numpy, scipy; \
	print('xgboost', xgboost.__version__); print('sklearn', sklearn.__version__); \
	print('pandas', pandas.__version__); print('numpy', numpy.__version__); \
	print('scipy', scipy.__version__)"

# Licence-gated, so the download is not run for you. It is scripted rather than manual:
# accepting the licence once on Kaggle is the gate, and the CLI does the rest.
data:
	@echo "The dataset is licence-gated. Accept the licence once on Kaggle, then run these."
	@echo "Requires a Kaggle account and ~/.kaggle/kaggle.json (mode 600)."
	@echo ""
	@echo "  D=ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
	@echo "  for f in HI-Small_Trans.csv HI-Small_Patterns.txt \\"
	@echo "           LI-Small_Trans.csv LI-Small_Patterns.txt; do \\"
	@echo "      kaggle datasets download -d \$$D -f \$$f -p data/raw; \\"
	@echo "  done"
	@echo ""
	@echo "LI-Small feeds the published pipeline. HI-Small feeds 'make hi-small'."
	@echo "Source: Altman et al., Community Data License Agreement. arXiv 2306.16424."

all: ingest features models economics monitoring report

ingest:
	$(PY) -m src.data

features:
	$(PY) -m src.features

models:
	$(PY) -m src.models

economics:
	$(PY) -m src.economics

monitoring:
	$(PY) -m src.monitoring

report:
	$(PY) -m src.report

# Reported beside the published split, not part of it, so deliberately outside `all`.
window-scan:
	$(PY) -m src.window_scan

components:
	$(PY) -m src.components

# The section 14 pass-through window sweep. Refits per window value and reports how the
# operating point moves, reported beside the split. The 24h row reproduces the published run.
sweep:
	$(PY) -m src.sweep

# The reduced analyst brief generator, G9. One brief per alerted account, into reports/briefs/.
brief:
	$(PY) -m src.brief

# The HI-Small development variant. Same pipeline, same settings, other file. It reads
# config/params_hi_small.yaml, which differs from the published file only in the data
# contract and the output directories, and it writes to reports/hi-small/ so the published
# figures are never in its path. `make all` produces the headline on LI-Small. See D37, D38.
hi-small:
	PARAMS_PATH=config/params_hi_small.yaml $(MAKE) all

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests
	$(BIN)/mypy src tests

clean:
	rm -rf data/interim/* reports/figures/* reports/report.md
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
