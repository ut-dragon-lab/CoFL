"""One model consumes dynamically batched requests from independent scene workers."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from threading import Condition, Thread


class RemoteInferenceError(RuntimeError):
    """An inference transport failure is not a navigation episode outcome."""


class BatchedInferenceService:
    """Keep accepting work while the sole inference thread runs a tensor batch.

    The coordinator submits at most one outstanding request per environment.
    Keys include the worker and its monotonically increasing request ID; queued
    and completed requests retain their identity until the coordinator drains
    them. No environment barrier or per-item neural forward is used here.
    """

    def __init__(self, policy, parameters, *, max_batch_size, batch_wait_ms):
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(batch_wait_ms, bool)
            or not isinstance(batch_wait_ms, (int, float))
            or not math.isfinite(batch_wait_ms)
            or batch_wait_ms < 0
        ):
            raise ValueError("batch_wait_ms must be finite and nonnegative")
        self.policy = policy
        self.parameters = parameters
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = float(batch_wait_ms)
        self._condition = Condition()
        self._pending = deque()
        self._completed = deque()
        self._keys = set()
        self._stopping = False
        self._error = None
        self._submitted = self._predictions = 0
        self._batches = Counter()
        self._queue_wait_s = self._inference_s = 0.0
        self._started = time.perf_counter()
        self._thread = Thread(target=self._run, name="cofl-shared-inference", daemon=True)
        self._thread.start()

    def _raise_error(self):
        if self._error is not None:
            name = type(self._error).__name__
            advice = (
                " Reduce --inference-batch-size or --workers before retrying."
                if isinstance(self._error, MemoryError) or name == "OutOfMemoryError"
                else ""
            )
            raise RuntimeError(
                f"Shared model inference failed: {name}: {self._error}; "
                "uncommitted episodes can be retried with --resume." + advice
            ) from self._error

    def submit(self, key, sample):
        with self._condition:
            self._raise_error()
            if self._stopping:
                raise RuntimeError("Inference service is closed")
            if key in self._keys:
                raise ValueError(f"Duplicate inference request: {key!r}")
            self._keys.add(key)
            self._pending.append((key, sample, time.perf_counter()))
            self._submitted += 1
            self._condition.notify()

    def _next_batch(self):
        with self._condition:
            while not self._pending and not self._stopping:
                self._condition.wait()
            if self._stopping:
                return None
            deadline = self._pending[0][2] + self.batch_wait_ms / 1000.0
            while len(self._pending) < self.max_batch_size and not self._stopping:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            if self._stopping:
                return None
            return [self._pending.popleft() for _ in range(min(len(self._pending), self.max_batch_size))]

    def _run(self):
        try:
            while (batch := self._next_batch()) is not None:
                started = time.perf_counter()
                waits = [started - item[2] for item in batch]
                predictions = self.policy.predict_fields_batch(
                    [item[1] for item in batch], self.parameters
                )
                seconds = time.perf_counter() - started
                if len(predictions) != len(batch):
                    raise RuntimeError("Shared policy returned a different number of predictions")
                completed = []
                for (key, _sample, _enqueued), result, waiting in zip(
                    batch, predictions, waits, strict=True
                ):
                    completed.append(
                        (
                            key,
                            dict(
                                result,
                                batch_size=len(batch),
                                queue_wait_ms=waiting * 1000.0,
                                batch_inference_ms=seconds * 1000.0,
                            ),
                        )
                    )
                with self._condition:
                    self._completed.extend(completed)
                    self._predictions += len(batch)
                    self._batches[len(batch)] += 1
                    self._queue_wait_s += sum(waits)
                    self._inference_s += seconds
        except BaseException as error:
            with self._condition:
                self._error = error
                self._pending.clear()
                self._condition.notify_all()

    def drain(self):
        with self._condition:
            self._raise_error()
            completed = list(self._completed)
            self._completed.clear()
            self._keys.difference_update(key for key, _ in completed)
            return completed

    def snapshot(self):
        with self._condition:
            return {
                "mode": "shared_model_dynamic_batch",
                "model_instances": 1,
                "max_batch_size": self.max_batch_size,
                "batch_wait_ms": self.batch_wait_ms,
                "requests_submitted": self._submitted,
                "predictions_completed": self._predictions,
                "batches": sum(self._batches.values()),
                "batch_size_histogram": dict(sorted(self._batches.items())),
                "mean_batch_size": self._predictions / max(1, sum(self._batches.values())),
                "mean_queue_wait_ms": self._queue_wait_s * 1000.0 / max(1, self._predictions),
                "batch_inference_seconds": self._inference_s,
                "service_wall_seconds": time.perf_counter() - self._started,
            }

    def close(self):
        with self._condition:
            self._stopping = True
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Shared inference did not stop; its model must remain allocated")


class RemotePolicy:
    """Send one observation, receive its field, then integrate on the worker CPU."""

    def __init__(self, connection, geometry, parameters, worker_id):
        self.connection = connection
        self.geometry = dict(geometry)
        self.parameters = parameters
        self.worker_id = worker_id
        self.active_inference_backend = "shared_model_dynamic_batch"
        self._request_id = 0
        self._acceleration = {}

    def configure_sensors(self, sensors):
        from .policy import configure_geometry_sensors

        self.geometry = configure_geometry_sensors(self.geometry, sensors)

    @property
    def acceleration_status(self):
        return self._acceleration

    def predict(self, image, depth, instruction, parameters):
        from .policy import finish_prediction

        request_id = self._request_id
        self._request_id += 1
        started = time.perf_counter()
        try:
            self.connection.send(
                {
                    "event": "predict",
                    "request_id": request_id,
                    "sample": {
                        "image": image,
                        "depth": depth,
                        "instruction": instruction,
                        "geometry": dict(self.geometry),
                        "profile": "ground_sector_v1",
                    },
                }
            )
            response = self.connection.recv()
        except (OSError, EOFError) as error:
            raise RemoteInferenceError("Shared inference connection closed") from error
        if response.get("event") != "prediction" or response.get("request_id") != request_id:
            raise RemoteInferenceError("Shared prediction does not match the pending request")
        fields = response["result"]
        result = finish_prediction(fields, self.geometry, parameters)
        result["inference_batch"] = {
            "request_id": request_id,
            "worker_id": self.worker_id,
            "size": fields["batch_size"],
            "queue_wait_ms": fields["queue_wait_ms"],
            "batch_inference_ms": fields["batch_inference_ms"],
            "roundtrip_ms": (time.perf_counter() - started) * 1000.0,
        }
        self._acceleration = {
            **fields.get("acceleration", {}),
            "inference_service": "shared_model_dynamic_batch",
            "last_batch": result["inference_batch"],
        }
        return result

    def close(self):
        pass  # The worker owns the pipe; the coordinator owns the model.
