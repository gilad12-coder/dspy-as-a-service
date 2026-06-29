# Managed-inference gateway image for registry/PaaS deploys (e.g. Railway).
#
# docker-compose bind-mounts ./config.yaml into the stock litellm image; a PaaS
# has no bind mount, so we bake the same config into the image instead. The base
# image's entrypoint is ``litellm``, so CMD supplies exactly the args compose
# passes as ``command:``. The proxy listens on a fixed 4000.
#
# ``--host ::`` binds IPv6 (dual-stack on Linux, so IPv4 still works): required
# for a private-network deploy where the only caller is the backend reaching it
# at ``http://<service>.railway.internal:4000`` — Railway's private network is
# IPv6-only. Harmless for the local compose / public setups, which also work
# over the dual-stack socket.
FROM ghcr.io/berriai/litellm:main-stable
COPY config.yaml /app/config.yaml
CMD ["--config", "/app/config.yaml", "--port", "4000", "--host", "::"]
