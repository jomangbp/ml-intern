# Contributing to ML Intern Local

Thanks for your interest in contributing.

## Development Setup

```bash
git clone git@github.com:jomangbp/ml-intern-local.git
cd ml-intern-local
uv sync
uv tool install -e .
```

## Branching

- Create feature branches from `main`
- Use clear branch names, e.g. `feat/provider-ui`, `fix/zai-routing`

## Commit Guidelines

- Keep commits focused and small
- Write descriptive commit messages
- Include tests or validation steps when possible

## Pull Requests

Please include:

1. What changed
2. Why it changed
3. How it was tested
4. Any migration notes

## Security & Secrets

Never commit:

- `.env` files
- API keys/tokens
- local workspace state under `.ml-intern/`
- training artifacts and large generated binaries

If you discover a vulnerability, see [SECURITY.md](./SECURITY.md).
