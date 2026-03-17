# Contributing to optisim

Thanks for contributing. `optisim` is intentionally small, typed, and easy to modify, so changes should preserve that character.

## Ground Rules

- Keep behavior deterministic unless randomness is explicitly required and controllable.
- Prefer clear Python over framework-heavy abstractions.
- Add or update tests for changes to planning, IK, simulation, validation, or robot modeling.
- Document public API changes with docstrings and user-facing docs when relevant.
- Avoid unrelated refactors in the same pull request.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,viz]
```

## Typical Workflow

1. Create a branch from `main`.
2. Make a focused change with tests.
3. Run `pytest -q`.
4. Run the affected example or CLI path when behavior changes.
5. Open a pull request with a concise description of the problem, approach, and validation.

## Code Style

- Follow the existing module layout and naming conventions.
- Keep functions and classes small enough to inspect quickly.
- Use type hints for public APIs.
- Add succinct docstrings to public classes and functions.
- Preserve compatibility with supported Python versions in CI.

## Tests

The test suite is the baseline quality gate:

```bash
pytest -q
```

If you change example behavior, also run one or more bundled examples:

```bash
python -m optisim validate examples/pick_and_place.yaml
python -m optisim run examples/pick_and_place.yaml --visualize
```

## Issues and Pull Requests

- Open an issue for bugs, feature requests, or design discussions when the change is non-trivial.
- Keep pull requests scoped to one problem or feature.
- Include before/after behavior when fixing a bug.
- Mention any follow-up work that remains intentionally out of scope.

## Documentation

README quality matters. If your change affects installation, examples, task authoring, robot definitions, or visualization, update the relevant docs in the same pull request.

## License

By contributing, you agree that your contributions will be licensed under the MIT License in this repository.
