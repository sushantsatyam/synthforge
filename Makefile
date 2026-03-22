.PHONY: install install-dev test lint format clean examples

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=synthforge --cov-report=html --tb=short

lint:
	ruff check synthforge/ tests/
	mypy synthforge/ --ignore-missing-imports

format:
	ruff format synthforge/ tests/

examples:
	python examples/sensor_data.py
	python examples/ecommerce_catalog.py
	python examples/hr_dataset.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
