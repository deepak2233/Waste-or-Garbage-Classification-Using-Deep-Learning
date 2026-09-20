"""HTTP serving. Requires the optional ``serve`` extra."""

__all__ = ["create_app"]


def __getattr__(name: str):
    # Deferred so that importing wasteclf.serving does not require FastAPI.
    if name == "create_app":
        from wasteclf.serving.app import create_app

        return create_app
    raise AttributeError(name)
