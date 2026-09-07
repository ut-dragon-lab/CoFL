"""Object and region categories used for instruction candidates."""

from __future__ import annotations

L1_NAV_OBJECTS: frozenset[str] = frozenset({"door", "stairs"})
L2_LANDMARK_OBJECTS: frozenset[str] = frozenset(
    {
        "table",
        "chair",
        "seating",
        "bed",
        "sink",
        "cabinet",
        "appliances",
        "plant",
        "counter",
        "shower",
        "mirror",
        "window",
        "shelving",
        "tv_monitor",
    }
)
OBJECT_SURFACE_FORMS: dict[str, tuple[str, ...]] = {
    "door": ("door", "doorway"),
    "stairs": ("stairs", "staircase", "stairway"),
    "table": ("table", "desk"),
    "chair": ("chair", "stool"),
    "seating": ("couch", "sofa", "bench"),
    "bed": ("bed",),
    "sink": ("sink", "vanity"),
    "cabinet": ("cabinet", "cupboard"),
    "appliances": ("refrigerator", "stove", "oven"),
    "plant": ("plant", "flowers"),
    "counter": ("counter", "countertop"),
    "shower": ("shower", "bathtub"),
    "mirror": ("mirror",),
    "window": ("window",),
    "shelving": ("bookshelf", "shelf"),
    "tv_monitor": ("tv", "television"),
}
ALL_ALLOWED_OBJECTS: frozenset[str] = L1_NAV_OBJECTS | L2_LANDMARK_OBJECTS


def canonical_object_noun(cat: str) -> str:
    forms = OBJECT_SURFACE_FORMS.get(cat)
    return forms[0] if forms else cat
