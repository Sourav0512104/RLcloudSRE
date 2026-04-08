import importlib
import asyncio


def test_connect_env_uses_async_docker_client(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    class DummyAsyncEnv:
        pass

    async_env = DummyAsyncEnv()

    async def fake_from_docker_image(image):
        assert image == "test-image"
        return async_env

    monkeypatch.setattr(inference, "IMAGE_NAME", "test-image")
    monkeypatch.setattr(inference, "ENV_BASE_URL", None)
    monkeypatch.setattr(inference.CloudSreRlEnv, "from_docker_image", fake_from_docker_image)

    assert asyncio.run(inference.connect_env()) is async_env


def test_connect_env_connects_direct_client(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    class DummyClient:
        def __init__(self, base_url):
            self.base_url = base_url
            self.connected = False

        async def connect(self):
            self.connected = True

    monkeypatch.setattr(inference, "IMAGE_NAME", None)
    monkeypatch.setattr(inference, "ENV_BASE_URL", "http://env.example")
    monkeypatch.setattr(inference, "CloudSreRlEnv", DummyClient)

    client = asyncio.run(inference.connect_env())

    assert client.base_url == "http://env.example"
    assert client.connected is True


def test_create_llm_client_uses_validator_env_vars(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    captured = {}

    class DummyOpenAI:
        def __init__(self, *, base_url, api_key):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("API_KEY", "validator-key")
    monkeypatch.setattr(inference, "OpenAI", DummyOpenAI)

    inference.create_llm_client()

    assert captured == {
        "base_url": "https://proxy.example/v1",
        "api_key": "validator-key",
    }


def test_warmup_llm_proxy_makes_chat_completion(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    captured = {}

    class DummyCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    inference.warmup_llm_proxy(DummyClient())

    assert captured["model"] == inference.MODEL_NAME
    assert captured["stream"] is False
