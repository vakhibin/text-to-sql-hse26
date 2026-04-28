.PHONY: compose-up compose-down compose-logs compose-ps compose-build compose-test

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

compose-ps:
	docker compose ps

compose-build:
	docker compose build

compose-test:
	docker compose run --rm text-to-sql-api uv run --no-sync pytest -q
