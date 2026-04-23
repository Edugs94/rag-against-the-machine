MODULE = src
VENV = .venv
VENV_FILE = $(VENV)/.pyinstall.timestamp
UV_FILES = pyproject.toml uv.lock
UV_RUN = uv run python3
ARGS ?= index

all: install run

install: $(VENV_FILE)

$(VENV_FILE): $(UV_FILES)
	@echo "Installing..."
	uv sync
	@touch $(VENV_FILE)

run: install
	@echo "Running..."
	$(UV_RUN) -m $(MODULE) $(ARGS)

debug: install
	@echo "Debugging..."
	$(UV_RUN) -m pdb -m $(MODULE) $(ARGS)

lint: install
	@echo "Linting..."
	@echo "Flake8: "
	$(UV_RUN) -m flake8 $(MODULE)
	@echo "Mypy: "
	$(UV_RUN) -m mypy $(MODULE) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache

fclean: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)

re: fclean all

test: install
	$(UV_RUN) -m pytest

.PHONY: all run debug lint lint-strict clean fclean re install test