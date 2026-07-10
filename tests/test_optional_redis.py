"""Tests for running VERA without the optional managed Redis cache."""

import asyncio

from api import main


def _stub_required_services(monkeypatch):
    monkeypatch.setattr(main, "AzureOpenAIService", lambda **kwargs: object())
    monkeypatch.setattr(main, "AzureSpeechService", lambda **kwargs: object())
    monkeypatch.setattr(main, "AzureSearchService", lambda **kwargs: object())


def test_initialize_services_without_redis(monkeypatch):
    _stub_required_services(monkeypatch)
    monkeypatch.delenv("REDIS_CONNECTION_STRING", raising=False)

    def unexpected_redis(**kwargs):
        raise AssertionError("Redis should not be initialized without a connection string")

    monkeypatch.setattr(main, "RedisCacheService", unexpected_redis)
    main.redis_cache = object()  # prove re-initialization clears stale state

    asyncio.run(main.initialize_azure_services())

    assert main.redis_cache is None
    assert main.azure_openai is not None
    assert main.azure_speech is not None
    assert main.azure_search is not None


def test_initialize_services_with_redis(monkeypatch):
    _stub_required_services(monkeypatch)
    monkeypatch.setenv("REDIS_CONNECTION_STRING", "rediss://example.invalid:6380")
    created = object()
    monkeypatch.setattr(main, "RedisCacheService", lambda **kwargs: created)

    asyncio.run(main.initialize_azure_services())

    assert main.redis_cache is created


def test_health_is_healthy_when_only_redis_is_disabled(monkeypatch):
    monkeypatch.setattr(main, "azure_openai", object())
    monkeypatch.setattr(main, "azure_speech", object())
    monkeypatch.setattr(main, "azure_search", object())
    monkeypatch.setattr(main, "redis_cache", None)

    response = asyncio.run(main.health_check())

    assert response.status == "healthy"
    assert response.services["redis"] == "disabled"


def test_health_is_degraded_when_required_service_is_disabled(monkeypatch):
    monkeypatch.setattr(main, "azure_openai", object())
    monkeypatch.setattr(main, "azure_speech", None)
    monkeypatch.setattr(main, "azure_search", object())
    monkeypatch.setattr(main, "redis_cache", None)

    response = asyncio.run(main.health_check())

    assert response.status == "degraded"
    assert response.services["azure_speech"] == "disabled"
