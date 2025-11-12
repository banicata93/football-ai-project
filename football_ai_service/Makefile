# Makefile за Football AI Service

.PHONY: help install install-dev clean test lint format run-api run-dev docker-build docker-run

# Показва помощ
help:
	@echo "Налични команди:"
	@echo "  install       - Инсталира зависимости"
	@echo "  install-dev   - Инсталира с development зависимости"
	@echo "  clean         - Почиства временни файлове"
	@echo "  test          - Пуска тестове"
	@echo "  lint          - Проверява код стил"
	@echo "  format        - Форматира кода"
	@echo "  run-api       - Стартира API сървъра"
	@echo "  run-dev       - Стартира в development режим"
	@echo "  docker-build  - Билдва Docker image"
	@echo "  docker-run    - Пуска Docker контейнер"

# Инсталация на зависимости
install:
	pip install -r requirements.txt

# Инсталация с development зависимости
install-dev:
	pip install -e ".[dev]"

# Почистване на временни файлове
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

# Тестове
test:
	python -m pytest tests/ -v

# Проверка на код стил
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Форматиране на код
format:
	black .
	isort .

# Стартиране на API сървъра
run-api:
	python api/main.py

# Стартиране в development режим
run-dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 3000

# Docker билд
docker-build:
	docker build -t football-ai-service .

# Docker пуск
docker-run:
	docker run -p 3000:3000 football-ai-service

# Генериране на features
generate-features:
	python pipelines/generate_features.py

# Тренировка на модели
train-models:
	python pipelines/train_poisson.py
	python pipelines/train_ml_models.py
	python pipelines/train_ensemble.py

# Тест на API
test-api:
	python api/test_api.py

# Всичко заедно (инсталация + тренировка + стартиране)
setup-all: install generate-features train-models run-api
