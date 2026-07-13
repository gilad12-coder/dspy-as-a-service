# Security Policy

## Supported versions

Skynet is pre-1.0 and ships from `main`. Security fixes land on `main` and in
the latest tagged release; older tags are not backported.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue, pull
request, or discussion for anything security-sensitive.

Preferred: use GitHub's private vulnerability reporting on this repository
(**Security → Report a vulnerability**). It keeps the report confidential and
threads the fix through a private advisory.

Alternatively, email **gilad.mo12@gmail.com** with:

- a description of the issue and its impact,
- steps to reproduce (a minimal proof-of-concept helps),
- the affected version or commit, and
- any suggested remediation.

We aim to acknowledge a report within five business days and to agree a
disclosure timeline with you. Please give us a reasonable window to ship a fix
before any public disclosure. Reporters who want credit will get it.

## Where to look

Skynet stores provider API keys and runs a bring-your-own-key (BYOK) secret
vault, so credential storage, authentication and session handling, tenant
isolation, and the optimization/serving runtime are the areas where a bug hurts
most. When in doubt, report it — we would rather triage a false alarm than miss
a real issue.
