"""Explicit policy device selection and CUDA failures, without eager Torch imports."""

from __future__ import annotations

import gc
import re
import sys
import traceback
import uuid
from contextlib import contextmanager
from threading import RLock


def normalize_device(device):
    if device == "cuda":
        return "cuda:0"
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("Policy device must be cpu or cuda:N")
    return device


class DeviceExecutionError(RuntimeError):
    """A policy CUDA failure already recorded by the shared device manager."""


def is_cuda_failure(error):
    if isinstance(error, DeviceExecutionError):
        return True
    if type(error).__name__ == "OutOfMemoryError" and type(error).__module__.startswith("torch"):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cuda error:",
            "cuda driver",
            "cuda initialization",
            "cuda unknown error",
            "nvidia driver on your system is too old",
            "cuda is not available",
            "cuda is unavailable",
            "invalid device ordinal",
            "not compiled with cuda",
            "no nvidia driver",
            "found no nvidia driver",
            "no cuda gpus are available",
            "cublas_status_",
            "cudnn_status_",
        )
    )


def clear_exception_frames(error):
    """Failed checkpoint transfers must not retain partial models through traceback frames."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
        error = error.__cause__ or error.__context__


def release_cuda(device):
    gc.collect()
    torch = sys.modules.get("torch")
    if not device.startswith("cuda:") or torch is None or not torch.cuda.is_initialized():
        return
    try:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    except Exception as error:
        if not is_cuda_failure(error):
            raise
        clear_exception_frames(error)


def cuda_capability(device):
    """Inspect the driver and device properties without allocating a model."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "gpu_name": None,
                "count": 0,
                "message": "CUDA is unavailable in this PyTorch runtime or NVIDIA driver",
            }
        count = torch.cuda.device_count()
        index = int(device.split(":")[1]) if device.startswith("cuda:") else 0
        if index >= count:
            return {
                "gpu_available": True,
                "gpu_name": None,
                "count": count,
                "message": f"Invalid device ordinal: {device}; detected {count} CUDA device(s)",
            }
        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(index),
            "count": count,
            "message": None,
        }
    except (ImportError, RuntimeError, AssertionError, OSError) as error:
        return {"gpu_available": False, "gpu_name": None, "count": 0, "message": str(error)}


class DeviceManager:
    def __init__(self, device="cuda:0", *, probe=None):
        self._device = normalize_device(device)
        self._probe = probe or cuda_capability
        self._lock = RLock()
        self._failure = None
        self._failure_origin = None
        self._generation = 0

    @property
    def device(self):
        with self._lock:
            return self._device

    def select(self, device):
        with self._lock:
            self._device = normalize_device(device)
            self._generation += 1
            self._failure = self._failure_origin = None

    def _record(self, device, message, *, origin):
        with self._lock:
            if (
                origin == "probe"
                and self._failure is not None
                and (
                    self._failure_origin == "execution"
                    or (self._failure["device"] == device and self._failure["message"] == message)
                )
            ):
                return
            self._failure = {"id": uuid.uuid4().hex, "device": device, "message": message}
            self._failure_origin = origin

    def validate(self, device=None):
        with self._lock:
            device = normalize_device(device or self._device)
            generation = self._generation
        if device == "cpu":
            return
        capability = self._probe(device)
        if not capability["gpu_available"] or capability.get("message"):
            message = capability.get("message") or "CUDA is unavailable"
            with self._lock:
                if generation == self._generation:
                    self._record(device, message, origin="probe")
            raise DeviceExecutionError(message)

    def status(self, *, can_change=True):
        while True:
            with self._lock:
                device, generation = self._device, self._generation
            capability = self._probe(device)
            with self._lock:
                if generation != self._generation:
                    continue
                if device.startswith("cuda:") and (
                    not capability["gpu_available"] or capability.get("message")
                ):
                    self._record(
                        device, capability.get("message") or "CUDA is unavailable", origin="probe"
                    )
                return {
                    "device": self._device,
                    "gpu_available": capability["gpu_available"],
                    "gpu_name": capability.get("gpu_name"),
                    "failure": None if self._failure is None else dict(self._failure),
                    "can_change": can_change,
                }

    @contextmanager
    def execution(self, *, check=True, cleanup=None):
        with self._lock:
            device, generation = self._device, self._generation
        try:
            if check:
                self.validate(device)
            yield
        except Exception as error:
            if not device.startswith("cuda:") or not is_cuda_failure(error):
                raise
            message = str(error)
            with self._lock:
                if generation == self._generation and not isinstance(error, DeviceExecutionError):
                    self._record(device, message, origin="execution")
            clear_exception_frames(error)
            try:
                if cleanup is not None:
                    cleanup()
                release_cuda(device)
            except Exception as cleanup_error:  # noqa: BLE001 — preserve the execution failure
                clear_exception_frames(cleanup_error)
            raise DeviceExecutionError(message) from None
        else:
            with self._lock:
                if generation == self._generation:
                    self._failure = self._failure_origin = None


def policy_device_operation(method):
    """Wrap a complete service call so failed frames can be released after unwinding."""
    from functools import wraps

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.device_manager.execution(check=False, cleanup=self.close):
            return method(self, *args, **kwargs)

    return wrapped
