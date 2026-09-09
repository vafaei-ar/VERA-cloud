"""Local integration fixture. NEVER deploy: external Azure services are disabled."""
from api import main


async def synthetic_services():
    # Guided dialog has no LLM requirement. TTS deliberately fails to exercise
    # the actual endpoint's text fallback, without Azure or clinical data.
    main.azure_openai = object()
    main.azure_search = object()
    main.azure_speech = object()
    main.redis_cache = None


main.initialize_azure_services = synthetic_services
main.config["caching"]["cache_warmup"] = False
app = main.app
