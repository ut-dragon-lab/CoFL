"""Closed-loop CoFL-S navigation, with an optional isolated Habitat runtime.

The public ``run_episode`` function is shared by the CLI and Studio. Importing
this package does not import Torch or any simulator libraries.
"""


def run_episode(*args, **kwargs):
    from .runner import run_episode as run

    return run(*args, **kwargs)


__all__ = ["run_episode"]
