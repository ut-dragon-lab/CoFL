"""Offline imports for the shared CoFL-series inference integrator.

Keep the established evaluation API while using exactly the production image
raster integration rules. Implementation lives in cofl.runtime.rollout.
"""

from cofl.runtime.rollout import RasterRollout, rollout_image_fields

__all__ = ["RasterRollout", "rollout_image_fields"]
