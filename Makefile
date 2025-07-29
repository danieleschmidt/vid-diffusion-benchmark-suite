.PHONY: help install install-dev test test-cov lint format type-check clean build docs serve-docs

help:				## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:			## Install package dependencies
	pip install -e .

install-dev:		## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:				## Run tests
	pytest

test-cov:			## Run tests with coverage
	pytest --cov=vid_diffusion_bench --cov-report=html --cov-report=term-missing

lint:				## Run linting
	ruff check src tests
	black --check src tests
	isort --check-only src tests

format:				## Format code
	black src tests
	isort src tests
	ruff check --fix src tests

type-check:			## Run type checking
	mypy src

security:			## Run security checks
	bandit -r src
	safety check
	detect-secrets scan --baseline .secrets.baseline

clean:				## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:				## Build package
	python -m build

docs:				## Build documentation
	mkdocs build

serve-docs:			## Serve documentation locally
	mkdocs serve

docker-build:		## Build Docker image
	docker build -t vid-diffusion-benchmark-suite .

docker-run:			## Run Docker container
	docker run -it --rm vid-diffusion-benchmark-suite