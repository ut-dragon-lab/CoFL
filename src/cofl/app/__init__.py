"""Optional browser application. Importing this package does not load a model."""


def create_app(*args, **kwargs):
    """Create a Studio ASGI application; requires the ``app`` installation extra."""
    from .server import create_app as factory

    return factory(*args, **kwargs)
