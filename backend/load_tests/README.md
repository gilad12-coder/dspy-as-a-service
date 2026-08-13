# Scale-readiness gate

This harness answers one narrow launch question: can the deployed Skynet
topology serve hundreds of active users across representative product journeys
without HTTP errors or unacceptable latency while real workers process a job
burst?

It runs entirely on local or GitHub-hosted infrastructure. The model and
embedding endpoints are deterministic OpenAI-compatible mocks, so the test
exercises the production Next.js server, API, authentication, Redis, PgBouncer,
pgvector, Postgres, workers, subprocesses, and SSE paths without spending
provider credits.

## Target scenario

`mixed_realistic` ramps 200 distinct authenticated users over 20 seconds, then
holds all users for a 60-second soak. In parallel it drives:

- `/run` and `/grid-search` submissions through the worker queue;
- dashboard list, counts, sidebar, and job-summary reads;
- dashboard analytics aggregation;
- dataset profile and split validation;
- model-catalog reads;
- semantic Explore search through the embedding gateway and pgvector;
- `/login` rendering and the NextAuth session endpoint on the production
  Next.js image; and
- 24 long-lived job progress streams.

Each user gets an independent browser-like connection pool, reuses one bearer
token, and pauses between actions. Forty-eight users submit one job each; every
other user remains read-heavy. The stack contains three HTTP-only API replicas,
three standalone worker replicas, PgBouncer in transaction mode, shared Redis,
pgvector-enabled Postgres, the production frontend image, and the mock
model/embedding service.

The release gate fails the process when any of these conditions is true:

- HTTP error rate is above 0.5%;
- p95 latency is above 1.5 seconds;
- p99 latency is above 3 seconds;
- throughput falls below 40 requests per second;
- any virtual user fails to become active;
- any submission or SSE connection is rejected;
- more than 5% of Explore searches fall back during transient indexing;
- no frontend request reaches the production Next.js server; or
- any submitted mock-model job remains non-terminal after a 120-second queue
  drain window, fails, or is cancelled.

JSON and Markdown reports are written to `load_tests/results/` and contain the
operation mix, queue state, SLO thresholds, and violations.

## Run it

From `backend/`:

```bash
.venv/bin/python -m load_tests.run_all --scenarios=mixed_realistic
```

For a faster smoke run:

```bash
LOAD_TEST_MIXED_USERS=20 \
LOAD_TEST_MIXED_SUBMIT_USERS=4 \
LOAD_TEST_MIXED_SSE_CONNECTIONS=2 \
LOAD_TEST_MIXED_RAMP_SECONDS=2 \
LOAD_TEST_MIXED_SOAK_SECONDS=5 \
.venv/bin/python -m load_tests.run_all --scenarios=mixed_realistic
```

The GitHub Actions **scale readiness** workflow runs the full target every
Monday and can be triggered manually before a launch. Its metrics and container
logs are retained as a workflow artifact.

## What a pass proves

A pass proves the application behavior, frontend HTTP delivery, semantic-search
path, and the tested topology at the resources available to the machine running
the harness. It does not prove that an unscaled production deployment has
equivalent CPU, RAM, database capacity, browser rendering performance on every
client device, or a third-party provider's live quota. Before launch,
production still needs the same topology and capacity assumptions: multiple API
replicas, separate worker replicas, PgBouncer, shared Redis, a configured
`/health` check, non-zero deployment draining time, and upstream model limits
sized from the measured job rate.
