"""Expose encrypted, provider-agnostic BYOK connection management."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...byok import ProviderKeyVault
from ..auth import AuthenticatedUser, get_authenticated_user
from ..model_catalog import (
    CatalogModel,
    CatalogProvider,
    ModelCatalogResponse,
    get_byok_catalog_cached,
)
from .models import discover_models_at_endpoint

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


# Masked provider connection returned to its owner.
class ProviderKeyResponse(BaseModel):
    id: str
    provider: str
    label: str | None = None
    last4: str
    api_base: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    status: str
    added_at: str


# Envelope for all of the caller's provider connections.
class ProviderKeysResponse(BaseModel):
    keys: list[ProviderKeyResponse] = Field(default_factory=list)


# Arbitrary OpenAI-compatible or LiteLLM connection supplied by a user.
class SaveProviderKeyRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=32)
    secret: str = Field(min_length=1)
    label: str | None = Field(default=None, max_length=120)
    api_base: str | None = Field(default=None, max_length=255)
    params: dict[str, Any] = Field(default_factory=dict)


def _response(view: Any) -> ProviderKeyResponse:
    """Project a vault view onto the HTTP response.

    Args:
        view: Masked provider connection from the vault.

    Returns:
        Secret-free provider response.
    """
    return ProviderKeyResponse(
        id=view.id,
        provider=view.provider,
        label=view.label,
        last4=view.last4,
        api_base=view.api_base,
        params=view.params,
        status=view.status,
        added_at=view.added_at,
    )


def _catalog_for_user(vault: ProviderKeyVault, username: str) -> ModelCatalogResponse:
    """Combine built-in models with discoverable user connection models.

    Args:
        vault: Encrypted provider connection vault.
        username: Account whose endpoints may be queried.

    Returns:
        Account-scoped model catalog.
    """
    base = get_byok_catalog_cached()
    providers = list(base.providers)
    models = list(base.models)
    provider_keys = {provider.slug for provider in providers}
    model_keys = {(model.provider, model.value) for model in models}
    for view in vault.list_keys(username).keys:
        if not view.api_base:
            continue
        resolved = vault.resolve_connection(username, view.provider)
        if resolved is None:
            continue
        discovered = discover_models_at_endpoint(view.api_base, resolved.secret)
        if discovered.error:
            continue
        if view.provider not in provider_keys:
            providers.append(
                CatalogProvider(
                    slug=view.provider,
                    label=view.label or view.provider,
                    default_base_url=view.api_base,
                    has_env_key=True,
                )
            )
            provider_keys.add(view.provider)
        for model_id in discovered.models:
            bare = model_id.strip().strip("/")
            if not bare:
                continue
            value = f"openai/{bare.removeprefix('openai/')}"
            if view.provider == "openrouter":
                value = f"openrouter/{bare.removeprefix('openrouter/')}"
            key = (view.provider, value)
            if key in model_keys:
                continue
            models.append(
                CatalogModel(
                    value=value,
                    label=bare,
                    provider=view.provider,
                    byok_provider=view.provider,
                    available=True,
                )
            )
            model_keys.add(key)
    return ModelCatalogResponse(providers=providers, models=models)


def create_byok_router(*, job_store: Any) -> APIRouter:
    """Build authenticated BYOK connection routes.

    Args:
        job_store: Store whose PostgreSQL engine persists encrypted secrets.

    Returns:
        Configured provider connection router.
    """
    router = APIRouter(prefix="/byok")
    vault = ProviderKeyVault(engine=job_store.engine)

    @router.get("/keys", response_model=ProviderKeysResponse)
    def list_provider_keys(user: AuthenticatedUserDep) -> ProviderKeysResponse:
        """List the caller's masked provider connections.

        Args:
            user: Authenticated connection owner.

        Returns:
            Stored connections without plaintext secrets.
        """
        return ProviderKeysResponse(keys=[_response(view) for view in vault.list_keys(user.username).keys])

    @router.get("/models", response_model=ModelCatalogResponse)
    def list_byok_models(user: AuthenticatedUserDep) -> ModelCatalogResponse:
        """List built-in and discoverable models for the caller's connections.

        Args:
            user: Authenticated connection owner.

        Returns:
            Account-scoped BYOK model catalog.
        """
        return _catalog_for_user(vault, user.username)

    @router.put("/keys", response_model=ProviderKeyResponse)
    def save_provider_key(
        body: SaveProviderKeyRequest,
        user: AuthenticatedUserDep,
    ) -> ProviderKeyResponse:
        """Encrypt and store an arbitrary provider connection.

        Args:
            body: Provider slug, secret, endpoint and pass-through parameters.
            user: Authenticated connection owner.

        Returns:
            Masked stored connection.
        """
        view = vault.save_key(
            user.username,
            body.provider,
            body.secret,
            label=body.label,
            api_base=body.api_base,
            params=body.params,
        )
        return _response(view)

    @router.post("/keys/{provider}/verify", response_model=ProviderKeyResponse)
    def verify_provider_key(
        provider: str,
        user: AuthenticatedUserDep,
    ) -> ProviderKeyResponse:
        """Probe a stored connection when its endpoint supports discovery.

        Args:
            provider: Stored provider slug.
            user: Authenticated connection owner.

        Returns:
            Masked connection with its latest verification state.
        """
        return _response(vault.verify_key(user.username, provider))

    @router.delete("/keys/{provider}", response_model=ProviderKeysResponse)
    def remove_provider_key(
        provider: str,
        user: AuthenticatedUserDep,
    ) -> ProviderKeysResponse:
        """Delete a provider connection and return the remaining list.

        Args:
            provider: Stored provider slug.
            user: Authenticated connection owner.

        Returns:
            Remaining masked connections.
        """
        vault.remove_key(user.username, provider)
        return ProviderKeysResponse(keys=[_response(view) for view in vault.list_keys(user.username).keys])

    return router
