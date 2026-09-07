"""Drain admitted API requests and close shared Studio resources exactly once."""

import asyncio
import inspect

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse


class ShutdownError(RuntimeError):
    """At least one resource could not confirm that its cleanup finished."""


class ShutdownController:
    def __init__(self, resources, *, callback=None):
        self._resources = dict(resources)
        self._finished = set()
        self._callback = callback
        self._callback_called = False
        self._callback_task = None
        self._stop_requested = False
        self._task = None
        self._active_requests = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self.closing = False
        self.closed = False

    @property
    def server_stopping(self):
        return self._callback is not None

    def admit(self):
        if self.closing:
            return False
        self._active_requests += 1
        self._drained.clear()
        return True

    def release(self):
        self._active_requests -= 1
        if self._active_requests == 0:
            self._drained.set()

    async def _cleanup(self):
        await self._drained.wait()
        failures = []
        for name, close in self._resources.items():
            if name in self._finished:
                continue
            try:
                await run_in_threadpool(close)
            except Exception as error:  # noqa: BLE001 — attempt every independent resource cleanup
                failures.append(f"{name}: {type(error).__name__}: {error}")
            else:
                self._finished.add(name)
        if failures:
            raise ShutdownError(
                "Studio cleanup is incomplete; retry closing. " + "; ".join(failures)
            )
        self.closed = True
        if self._stop_requested:
            await self.stop_server()

    async def shutdown(self, *, request_server_stop=False):
        self._stop_requested = self._stop_requested or request_server_stop
        self.closing = True
        if self.closed:
            if self._stop_requested:
                await self.stop_server()
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._cleanup(), name="CoFLStudioShutdown")
            # A disconnected caller may no longer consume the task's exception.
            self._task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        await asyncio.shield(self._task)

    async def stop_server(self):
        if not self.closed or self._callback is None or self._callback_called:
            return
        if self._callback_task is None or self._callback_task.done():
            self._callback_task = asyncio.create_task(self._invoke_callback())
        await asyncio.shield(self._callback_task)

    async def _invoke_callback(self):
        try:
            result = self._callback()
            if inspect.isawaitable(result):
                await result
        except Exception as error:
            raise ShutdownError(
                f"Resources closed, but requesting server exit failed: {error}"
            ) from error
        self._callback_called = True


class ShutdownMiddleware:
    """Hold admission through response completion; the shutdown endpoint drains these leases."""

    def __init__(self, app, controller):
        self.app = app
        self.controller = controller

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        guarded = (
            scope["type"] == "http"
            and path.startswith("/api/")
            and path != "/api/v1/runtime/shutdown"
        )
        if not guarded:
            await self.app(scope, receive, send)
            return
        if not self.controller.admit():
            response = JSONResponse(
                {"detail": "Studio is closing or already closed"}, status_code=503
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.controller.release()
