import importlib


def test_connect_env_wraps_async_docker_client(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    class DummySyncEnv:
        pass

    sync_env = DummySyncEnv()

    class DummyAsyncEnv:
        def sync(self):
            return sync_env

    async def fake_from_docker_image(image):
        assert image == "test-image"
        return DummyAsyncEnv()

    monkeypatch.setattr(inference, "IMAGE_NAME", "test-image")
    monkeypatch.setattr(inference, "ENV_BASE_URL", None)
    monkeypatch.setattr(inference.CloudSreRlEnv, "from_docker_image", fake_from_docker_image)

    assert inference.connect_env() is sync_env


def test_connect_env_wraps_direct_client(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    class DummySyncEnv:
        pass

    sync_env = DummySyncEnv()

    class DummyClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def sync(self):
            return sync_env

    monkeypatch.setattr(inference, "IMAGE_NAME", None)
    monkeypatch.setattr(inference, "ENV_BASE_URL", "http://env.example")
    monkeypatch.setattr(inference, "CloudSreRlEnv", DummyClient)

    assert inference.connect_env() is sync_env


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
