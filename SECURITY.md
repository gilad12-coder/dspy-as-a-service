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

## Known advisories with no available fix

Two open Dependabot alerts cannot currently be closed by upgrading. Both are
tracked here rather than silently dismissed, and both should be revisited when
upstream ships a fix.

**`diskcache` — CVE-2025-69872, unsafe pickle deserialization (medium).**
Affects `<= 5.6.3`; 5.6.3 is the newest release (August 2023) and no patched
version exists, so there is nothing to upgrade to. It reaches us through DSPy,
which uses it for the LM response cache — and the job worker deliberately runs
disk-only caching (`enable_memory_cache=False` in
`core/worker/subprocess_runner.py`) to keep a multi-hour run's resident memory
bounded, so the cache is live rather than dormant. Exploiting it requires
writing crafted data into the container's cache directory, which already
implies filesystem access to the worker. Revisit if DSPy swaps the cache
backend or diskcache resumes releases.

**`brace-expansion` — GHSA-mh99-v99m-4gvg, DoS via unbounded expansion (high).**
A `devDependency` only: it arrives via `minimatch@3.x` under the ESLint
toolchain (`eslint-plugin-react`, `@eslint/config-array`), runs at lint time on
our own source, and is never shipped. There is no clean fix today — forcing
`brace-expansion >= 5` breaks `minimatch@3.x` at runtime
(`TypeError: expand is not a function`, since the 5.x export is no longer a
callable), and every released ESLint still resolves the vulnerable chain. The
only remediation npm offers is downgrading `eslint-plugin-react` to 7.22.0.

For contrast, `sharp`'s libvips advisories *are* fixed: `next` caps it at
`^0.34.5`, so `frontend/package.json` carries a `sharp` override pinning
`^0.35.3`. Drop the override once Next widens the range.

## Where to look

Skynet stores provider API keys and runs a bring-your-own-key (BYOK) secret
vault, so credential storage, authentication and session handling, tenant
isolation, and the optimization/serving runtime are the areas where a bug hurts
most. When in doubt, report it — we would rather triage a false alarm than miss
a real issue.
