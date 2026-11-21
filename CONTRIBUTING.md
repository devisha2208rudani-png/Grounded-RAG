# Contributing

Thanks for contributing.

## Development Setup

1. Create a virtual environment.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the project in editable mode.

   ```bash
   pip install -e .
   ```

3. Run the test suite.

   ```bash
   python -m unittest discover -s tests -v
   ```

## Project Guidelines

- Keep the pipeline modular and interview-friendly.
- Prefer small, focused changes over broad rewrites.
- Preserve grounded answer behavior, citations, and fallback safety.
- Add or update tests when retrieval or generation behavior changes.

## Pull Requests

- Explain the motivation and trade-offs.
- Include test results.
- Update the README when user-facing behavior changes.

