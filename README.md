# text-to-sql-hse26
 Master's thesis that tries to solve text-to-sql problem by applying modern approaches

## Documentation

- [baseline.md](baseline.md) — архитектура агента (schema linking, SQL generator, pipeline) и описание компонентов системы
- [results.md](results.md) — ход экспериментов, результаты оценки на Spider dev set, анализ ошибок и выводы 

## Installation
1. Clone repository. For example
```commandline
git clone git@github.com:vakhibin/text-to-sql-hse26.git
```

2.  We manage environments and dependencies with UV. That's why one need to install UV first.

3. Install virtual environment and dependencies.
```commandline
uv venv 
uv sync
```

4. Run redis container via docker to enable caching
```commandline
docker run -d --name redis -p 6379:6379 redis:<version>
```

5. For using CLI it is highly recommended to set `PYTHONPATH` variable and add project root:
```commandline
export PYTHONPATH="${PYTHONPATH}:/home/user/Projects/text-to-sql-hse26"
```