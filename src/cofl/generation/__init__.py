"""Real data generation with lazy method and simulator dependencies."""

__all__ = [
    "GenerationConfig",
    "GenerationPipeline",
    "GenerationUnit",
    "create_pipeline",
    "generate_dataset",
    "load_mapping",
    "register_pipeline",
]

_FACTORIES = {}


def __getattr__(name):
    # The simulator worker can run in an older Habitat interpreter. Importing
    # this namespace must not load the host's native storage or typing stack.
    if name in {"GenerationConfig", "load_mapping"}:
        from . import config

        return getattr(config, name)
    if name in {"GenerationPipeline", "GenerationUnit"}:
        from . import types

        return getattr(types, name)
    raise AttributeError(name)


def register_pipeline(name, factory):
    """Register an external adapter without changing the shared runner.

    New coordinate representations also need an explicit dataset profile and
    validator; registering a generator never bypasses that storage contract.
    """
    if not isinstance(name, str) or not name.strip() or not callable(factory):
        raise ValueError("A pipeline registration needs a nonempty name and callable factory")
    if name in _FACTORIES or name in {"cofl", "cofl-s"}:
        raise ValueError(f"Pipeline {name!r} is already registered")
    _FACTORIES[name] = factory


def create_pipeline(config, *, base_dir):
    if config.method == "cofl":
        from .image import ImagePipeline

        factory = ImagePipeline
    elif config.method == "cofl-s":
        from .sector import SectorPipeline

        factory = SectorPipeline
    else:
        try:
            factory = _FACTORIES[config.method]
        except KeyError as error:
            raise ValueError(f"Unknown generation method: {config.method}") from error
    return factory({**config.pipeline, "split": config.split}, base_dir=base_dir)


def generate_dataset(config, output, *, base_dir=None, resume=False, pipeline=None):
    from .runner import generate_dataset as run

    return run(config, output, base_dir=base_dir, resume=resume, pipeline=pipeline)
