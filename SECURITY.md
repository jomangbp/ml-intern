# Security Policy

Thank you for helping keep **ML Intern Local** secure.

## Supported Versions

Security fixes are currently applied to the `main` branch.

## Reporting a Vulnerability

Please **do not** open public issues for security vulnerabilities.

Instead, report privately:

- Open a private security advisory on GitHub (preferred), or
- Contact the maintainer directly through GitHub.

When reporting, include:

1. A clear description of the issue
2. Reproduction steps
3. Impact assessment
4. Suggested fix (if available)

## Secrets and Credentials

Before opening issues or PRs, remove/redact:

- API keys and tokens
- `.env` contents
- Personal paths/usernames where possible
- Logs containing request headers or auth data

## Local Workspace Credentials

Provider keys saved from the UI are persisted in a per-user workspace `.env` file (outside tracked source files). Ensure this file is never committed.
