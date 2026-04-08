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


def test_require_api_key_uses_validator_variable(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    monkeypatch.setattr(inference, "API_KEY", "validator-key")

    assert inference.require_api_key() == "validator-key"


def test_require_api_key_allows_hf_token_fallback(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    monkeypatch.setattr(inference, "API_KEY", "hf-token-value")

    assert inference.require_api_key() == "hf-token-value"


def test_require_api_base_url_requires_validator_proxy(monkeypatch):
    inference = importlib.import_module("cloud_sre_rl.inference")

    monkeypatch.setattr(inference, "API_BASE_URL", "https://proxy.example/v1")

    assert inference.require_api_base_url() == "https://proxy.example/v1"
