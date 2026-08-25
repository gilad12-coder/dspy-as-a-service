# On-prem mirrors / artifactory: see AIRGAP.html.
#
# Build args:
#   BASE_IMAGE   — node:20-alpine compatible image; override with internal mirror.
#   NPM_REGISTRY — npm registry URL; override with Artifactory for air-gap builds.
#                  Note: lockfile `resolved` URLs must also point at the mirror,
#                  otherwise `npm ci` still fetches tarballs from the original host.
ARG BASE_IMAGE=node:20-alpine
ARG NPM_REGISTRY=https://registry.npmjs.org/

FROM ${BASE_IMAGE} AS base
ENV NEXT_TELEMETRY_DISABLED=1

FROM base AS deps
ARG NPM_REGISTRY
WORKDIR /app

COPY package.json package-lock.json .npmrc* ./
RUN npm config set registry "$NPM_REGISTRY" \
    && npm config set replace-registry-host always \
    && npm ci

FROM base AS build
WORKDIR /app
ENV NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Baked at build time for air-gap correctness: Next.js inlines these NEXT_PUBLIC_*
# values at `npm run build`, so they cannot be overridden on a prebuilt image.
ARG NEXT_PUBLIC_SITE_URL
ARG NEXT_PUBLIC_FEATURE_GENERALIST_AGENT
ENV NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL
ENV NEXT_PUBLIC_FEATURE_GENERALIST_AGENT=$NEXT_PUBLIC_FEATURE_GENERALIST_AGENT
RUN npm run build

FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3001

# Use UID 1001 with GID 0 — works everywhere (Docker, Podman, OpenShift restricted-v2 SCC).
# OpenShift assigns a random UID at runtime but always with GID 0, so runtime files must be
# group-0-owned and group-writable. Alpine uses busybox addgroup/adduser syntax.
RUN addgroup -g 1001 appuser 2>/dev/null || true \
    && adduser -u 1001 -G root -h /home/appuser -s /sbin/nologin -D appuser

COPY --from=build --chown=1001:0 /app/public ./public
COPY --from=build --chown=1001:0 /app/.next/standalone ./
COPY --from=build --chown=1001:0 /app/.next/static ./.next/static

RUN chgrp -R 0 /app && chmod -R g=u /app

USER 1001

EXPOSE 3001

CMD ["node", "server.js"]
