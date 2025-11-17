# Security Policy

## Supported Versions

This project is under active development. At this stage:

- The **default branch** (usually `main` or `master`) is treated as the supported line.
- Security fixes are applied to the default branch and released as soon as reasonably possible.
- Once the project reaches a first public release, we intend to follow **semantic versioning (SemVer)** for tags and published versions.

Older tags or branches may not receive backported security fixes unless explicitly stated.

## Reporting a Vulnerability

If you discover a security vulnerability in PriceSentinel:

1. **Do not open a public issue.**
2. Email the maintainer directly at: **joaosoarex@gmail.com**
   - Include a clear description of the issue and its potential impact.
   - Provide steps to reproduce, if possible.
   - Mention any relevant environment details (OS, Python version, configuration).

You can optionally encrypt your report if you prefer, but a plaintext email is acceptable for this project at its current stage.

We aim to:

- Acknowledge the report within a reasonable timeframe.
- Investigate and confirm the issue.
- Work on a fix and coordinate a release.
- Credit you appropriately in release notes, if you wish.

## Scope

This policy covers:

- The PriceSentinel code in this repository.
- The CLI and pipeline behavior when run with typical configurations.
- Handling of API keys and external services used by the project.

Third‑party services (e.g., ENTSO‑E, Open‑Meteo) have their own security posture; vulnerabilities in those services should be reported to their respective maintainers.

## Best Practices for Users

To reduce security risk when using PriceSentinel:

- Keep your Python environment and dependencies up to date.
- Store API keys in `.env` or environment variables, not in source control.
- Restrict permissions on data and log directories if they contain sensitive information.
- Review the project’s configuration files (`config/countries/*.yaml`) before deploying.
