# Contributing

Contributions that improve correctness, reproducibility, or usability are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff check .
mypy src/medical_imaging_qa
pytest
```

Please add tests for behavioural changes and avoid committing clinical or identifiable data. Synthetic fixtures are preferred.
