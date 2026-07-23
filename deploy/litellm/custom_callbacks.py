"""Proxy-side served-model passthrough for OpenRouter's Auto Router.

The backend reveals the concrete model behind an auto-routed turn by reading
the ``model`` field OpenRouter stamps on its raw SSE chunks (see
``core.service_gateway.language_models.install_openrouter_served_model_patch``).
This proxy is itself LiteLLM, which erases that field twice on the way out:

1. ``CustomStreamWrapper`` rewrites every chunk's ``model`` to the requested id
   (only Azure's model router is special-cased to keep the provider's answer).
2. ``proxy_server._restamp_streaming_chunk_model`` then re-stamps outgoing
   chunks with the client-requested model so internal deployment ids never
   leak — again with an Azure-router carve-out but none for OpenRouter.

Importing this module patches both spots: chunks from OpenRouter adopt the
provider-reported model, and the restamp skips requests aimed at OpenRouter's
Auto Router (``…openrouter/auto*``) — for those, naming the concrete pick is
the point, exactly like Azure's model router. Explicit (non-auto) requests
keep the stock restamp behavior. Side benefit: the proxy's own spend logs
record the concrete served model instead of the unpriced router group.

LiteLLM has no "run code at startup" hook, so the patches ride the callback
loader: ``config.yaml`` registers ``custom_callbacks.proxy_handler_instance``,
importing this module — the callback object itself is a no-op.
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

try:
    from litellm.proxy import proxy_server
except ImportError:  # pragma: no cover - depends on litellm internals
    proxy_server = None

_PATCH_FLAG = "_skynet_served_model_patch"


def _install_chunk_adoption() -> None:
    """Make streamed chunks keep the model OpenRouter actually served. Idempotent."""
    handler = CustomStreamWrapper.handle_openai_chat_completion_chunk
    if getattr(handler, _PATCH_FLAG, False):
        return

    def _adopting_handler(self, chunk):
        """Adopt the provider-reported model before normal chunk handling."""
        served = getattr(chunk, "model", None)
        if served and getattr(self, "custom_llm_provider", None) == "openrouter":
            self.model = served
        return handler(self, chunk)

    setattr(_adopting_handler, _PATCH_FLAG, True)
    CustomStreamWrapper.handle_openai_chat_completion_chunk = _adopting_handler
    print("skynet: served-model chunk adoption installed", flush=True)


def _install_restamp_carveout() -> None:
    """Keep the served model on chunks answering an Auto Router request. Idempotent."""
    original = getattr(proxy_server, "_restamp_streaming_chunk_model", None) if proxy_server else None
    if original is None:
        print("skynet: restamp hook not found; auto-route reveal limited to non-proxy path", flush=True)
        return
    if getattr(original, _PATCH_FLAG, False):
        return

    def _restamp_preserving_auto_route(
        *,
        chunk,
        requested_model_from_client,
        request_data,
        model_mismatch_logged,
        fallback_was_attempted=False,
        fallback_model_from_metadata=None,
    ):
        """Skip the restamp for auto-routed requests; defer to stock behavior otherwise."""
        if not fallback_was_attempted and "openrouter/auto" in (requested_model_from_client or ""):
            return chunk, model_mismatch_logged
        return original(
            chunk=chunk,
            requested_model_from_client=requested_model_from_client,
            request_data=request_data,
            model_mismatch_logged=model_mismatch_logged,
            fallback_was_attempted=fallback_was_attempted,
            fallback_model_from_metadata=fallback_model_from_metadata,
        )

    setattr(_restamp_preserving_auto_route, _PATCH_FLAG, True)
    proxy_server._restamp_streaming_chunk_model = _restamp_preserving_auto_route
    print("skynet: auto-route restamp carve-out installed", flush=True)


_install_chunk_adoption()
_install_restamp_carveout()


class _ServedModelPatch(CustomLogger):
    """No-op callback — exists only so the loader imports this module."""


proxy_handler_instance = _ServedModelPatch()
