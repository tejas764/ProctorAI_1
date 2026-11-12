# Contributing to ProctorAI

Thank you for your interest in improving ProctorAI, a research-oriented exam monitoring prototype.

## Getting started

1. Read the [README](README.md) for the architecture, prerequisites, and setup instructions.
2. Open an issue before undertaking a large change, describing the problem or proposed improvement.
3. Fork the repository or create a topic branch; keep pull requests focused.
4. Set up a Python virtual environment and install dependencies using `pip install -r requirement.txt`.
5. Run relevant checks and tests, and document what you verified.

## Areas to contribute

- Documentation, setup guides, and example configurations
- Reliability, testing, and performance of audio/video pipelines
- Accessible user interfaces and clearer error messages
- Privacy protections, data retention controls, and auditable monitoring workflows

## Pull request checklist

- [ ] The change has a clear purpose and a concise description
- [ ] Existing behavior is preserved or breaking changes are explained
- [ ] Relevant tests are added or manual verification is documented
- [ ] Documentation is updated when necessary
- [ ] No secrets, personal data, recordings, enrollment samples, or generated logs are included

## Privacy and responsible use

ProctorAI handles sensitive audio and video signals. Do not submit genuine student recordings, biometric data, exam footage, personal identifiers, or databases in issues or pull requests. Use synthetic data or properly consented test samples. Monitoring signals are probabilistic and should not be described as proof of misconduct or used for automatic disciplinary decisions.

## Reporting vulnerabilities

Follow [SECURITY.md](SECURITY.md) for responsible disclosure. Do not create a public issue containing exploitable details or sensitive data.

## Collaboration

Be respectful, constructive, and specific when discussing code, research results, or design choices.
