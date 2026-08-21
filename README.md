# mlops-practitioner
## Development Workflow

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Run linting and formatting checks:

```bash
ruff check src tests && black --check src tests
```

Run the test suite with coverage:

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

Train the model:

```bash
python -m prodml.train
```

Start the API server:

```bash
uvicorn prodml.api.main:app --reload --port 8000
```
