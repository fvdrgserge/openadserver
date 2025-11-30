# LiteAds Makefile
# Common commands for development and deployment

.PHONY: help install dev test lint format clean docker-up docker-down docker-logs

# Default target
help:
	@echo "LiteAds - Makefile Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install     Install dependencies"
	@echo "  make dev         Run development server"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linting"
	@echo "  make format      Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up   Start all services"
	@echo "  make docker-down Stop all services"
	@echo "  make docker-logs View service logs"
	@echo "  make docker-build Build Docker images"
	@echo ""
	@echo "Database:"
	@echo "  make db-init     Initialize database"
	@echo "  make db-migrate  Run migrations"
	@echo "  make db-mock     Generate mock data"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       Clean build artifacts"
	@echo "  make benchmark   Run performance benchmark"

# =============================================================================
# Development
# =============================================================================

install:
	pip install -e ".[dev]"

dev:
	LITEADS_ENV=dev python -m liteads.ad_server.main

test:
	pytest tests/ -v --cov=liteads --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --tb=short

lint:
	ruff check liteads tests
	mypy liteads --ignore-missing-imports

format:
	ruff format liteads tests
	ruff check --fix liteads tests

# =============================================================================
# Docker
# =============================================================================

docker-up:
	docker compose up -d

docker-up-full:
	docker compose --profile monitoring --profile production up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f ad-server

docker-build:
	docker compose build --no-cache

docker-restart:
	docker compose restart ad-server

docker-scale:
	docker compose up -d --scale ad-server=3

# =============================================================================
# Database
# =============================================================================

db-init:
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL to start..."
	@sleep 5
	docker compose exec postgres psql -U liteads -d liteads -f /docker-entrypoint-initdb.d/init.sql

db-migrate:
	alembic upgrade head

db-mock:
	python scripts/generate_mock_data.py --advertisers 10 --campaigns 5 --creatives 3

db-shell:
	docker compose exec postgres psql -U liteads -d liteads

# =============================================================================
# Redis
# =============================================================================

redis-cli:
	docker compose exec redis redis-cli

redis-flush:
	docker compose exec redis redis-cli FLUSHALL

# =============================================================================
# Utilities
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

benchmark:
	@echo "Running benchmark with wrk..."
	@echo "Make sure wrk is installed: brew install wrk"
	wrk -t4 -c100 -d30s -s scripts/benchmark.lua http://localhost:8000/api/v1/ad/request

health:
	curl -s http://localhost:8000/health | python -m json.tool

api-docs:
	@echo "API documentation available at:"
	@echo "  Swagger UI: http://localhost:8000/docs"
	@echo "  ReDoc:      http://localhost:8000/redoc"

# =============================================================================
# Production
# =============================================================================

prod-deploy:
	docker compose --profile production up -d --build

prod-logs:
	docker compose logs -f --tail=100

prod-status:
	docker compose ps
	@echo ""
	@echo "Health check:"
	@curl -s http://localhost/health | python -m json.tool || echo "Service not responding"
