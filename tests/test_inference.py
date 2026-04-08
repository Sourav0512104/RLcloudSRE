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
