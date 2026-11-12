# Security Policy

ProctorAI is a research and academic prototype involving camera, microphone, and biometric signals. It is **not** a production-certified proctoring or identity-verification service.

## Supported versions

Security maintenance is focused on the current `main` branch. Historical copies, experiments, and legacy reference modules may not receive security updates.

## Reporting a vulnerability

Please **do not disclose vulnerabilities publicly** before they can be assessed.

Where GitHub private vulnerability reporting is enabled, use the repository's **Security → Report a vulnerability** feature. Otherwise, contact the repository maintainer through their GitHub profile to agree on a private reporting channel. Do not include personal, biometric, or exam data in initial reports.

Useful information:
- Affected module and version/commit
- A description of the impact and conditions
- Reproduction steps using synthetic or non-sensitive sample data
- Suggested mitigation, if known

No fixed response or remediation SLA is promised.

## Sensitive data

Do not commit or share:
- Voice samples, face images, video frames, or exam recordings
- Enrollment profiles, biometric embeddings, or personal identifiers
- Local databases, monitoring logs, and generated evidence files
- Secrets, API keys, passwords, or production configuration

Users deploying or testing ProctorAI should explicitly obtain consent, limit retention, restrict access to recordings, and follow applicable privacy and data-protection requirements.

## Deployment guidance

- Run local development services on trusted networks only.
- Add authentication, authorization, transport encryption, and audit trails before any external deployment.
- Use isolated test data and protect access to administrative endpoints.
- Treat alerts as signals for review, **not** conclusive evidence of cheating.
- Review third-party dependencies and model licenses before distribution.

## Scope

This document provides a disclosure path and responsible-use guidance; it does not certify the project as secure or ready for live examinations.
