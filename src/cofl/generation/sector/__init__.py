"""CoFL-S generation; Habitat and native storage are loaded only when needed."""


def __getattr__(name):
    if name == "SectorPipeline":
        from .pipeline import SectorPipeline

        return SectorPipeline
    raise AttributeError(name)


__all__ = ["SectorPipeline"]
