# Contributing

Thanks for your interest in contributing to Highest Volatility! This guide covers the
local setup, code quality expectations, and how to prepare changes for review.

## Environment setup

1. Ensure you are using Python 3.10 or newer (the CI currently pins to 3.11).
2. Install the project in editable mode with the development extras:
   ```bash
   pip install -e .[dev]
   ```
   This command installs the runtime dependencies along with the developer
   toolchain (pytest, pytest-asyncio, pytest-cov, vcrpy, ruff, mypy, and
   hypothesis).
3. Optionally install any additional tooling needed for your workflow (for
   example, Redis when exercising cache integrations).

## Code quality and style

The repository relies on the following tools:

- **Ruff** for linting and formatting: `ruff check .` and `ruff format .`
- **Mypy** for static type checking: `mypy .`
- **Pytest** for the automated test suite: `pytest`

Run these commands locally before opening a pull request. Align code with the
existing style and follow PEP 8, the Zen of Python, and the design principles
noted in the project guidelines.

> **Note:** There is currently no shared `.pre-commit-config.yaml`. If you use
> pre-commit locally, please ensure it invokes the commands listed above.

## Pull request process

1. Create focused branches with cohesive changes.
2. Keep commits logical and well-described; reference related issues where
   applicable.
3. Ensure `ruff`, `mypy`, and `pytest` all pass locally. Include additional
   checks relevant to your change (e.g., benchmarking scripts) when applicable.
4. Update documentation, changelog entries, and tests as needed.
5. Push your branch and open a pull request. Describe the motivation, key
   changes, and any follow-up work in the PR description.
6. GitHub Actions workflows install dependencies via `pip install -e .[dev]`
   and execute automated jobs (e.g., benchmarks). Your PR must pass the CI
   pipeline before merge.
7. Address review feedback promptly; follow-up commits should keep the build
   green.

## Code of conduct

A dedicated code of conduct has not been published in this repository yet. Be
respectful and considerate when collaborating, and raise any conduct concerns to
project maintainers.
