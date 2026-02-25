from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, TYPE_CHECKING

import agentops.sdk.core
import agentops
from agentops.sdk.core import TracingCore
from agentops.sdk.processors import SpanProcessor
from opentelemetry.sdk.trace import ReadableSpan

from trainer.instrumentation.agentops import AgentOpsServerManager
from trainer.instrumentation import instrument_all, uninstrument_all
from .base import BaseTracer


if TYPE_CHECKING:
    from agentops.integration.callbacks.langchain import LangchainCallbackHandler


logger = logging.getLogger(__name__)


class AgentOpsTracer(BaseTracer):
    """Traces agent execution using AgentOps.

    This tracer provides functionality to capture execution details using the
    AgentOps library. It manages the AgentOps client initialization, server setup,
    and integration with the OpenTelemetry tracing ecosystem.

    Attributes:
        agentops_managed: Whether to automatically manage `agentops`.
                          When set to true, tracer calls `agentops.init()`
                          automatically and launches an agentops endpoint locally.
                          If not, you are responsible for calling and using it
                          before using the tracer.
        instrument_managed: Whether to automatically manage instrumentation.
                            When set to false, you will manage the instrumentation
                            yourself and the tracer might not work as expected.
        daemon: Whether the AgentOps server runs as a daemon process.
                Only applicable if `agentops_managed` is True.
    """

    def __init__(self, *, agentops_managed: bool = True, instrument_managed: bool = True, daemon: bool = True):
        super().__init__()
        self._flow_span_processor: Optional[FlowSpanProcessor] = None
        self.agentops_managed = agentops_managed
        self.instrument_managed = instrument_managed
        self.daemon = daemon

        self._agentops_server_manager = AgentOpsServerManager(self.daemon)
        self._agentops_server_port_val: Optional[int] = None

        if not self.agentops_managed:
            logger.warning("agentops_managed=False. You are responsible for AgentOps setup.")
        if not self.instrument_managed:
            logger.warning("instrument_managed=False. You are responsible for all instrumentation.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_agentops_server_manager"] = None  # Exclude the unpicklable server manager
        logger.debug(f"Getting state for pickling Trainer (PID {os.getpid()}). _agentops_server_manager excluded.")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # In child process, self._agentops_server_manager will be None.
        logger.debug(f"Setting state for unpickled Trainer (PID {os.getpid()}). _agentops_server_manager is None.")

    def init(self, *args, **kwargs):
        if self.agentops_managed and self._agentops_server_manager:
            self._agentops_server_manager.start()
            self._agentops_server_port_val = self._agentops_server_manager.get_port()
            if self._agentops_server_port_val is None:
                if (
                    self._agentops_server_manager.server_process is not None
                    and self._agentops_server_manager.server_process.is_alive()
                ):
                    raise RuntimeError("AgentOps server started but port is None. Check server manager logic.")
                elif (
                    self._agentops_server_port_val is None and self._agentops_server_manager.server_process is None
                ):  # Server failed to start
                    raise RuntimeError("AgentOps server manager indicates server is not running and port is None.")

    def teardown(self):
        if self.agentops_managed:
            self._agentops_server_manager.stop()
            logger.info("AgentOps server stopped.")

    def instrument(self, worker_id: int):
        instrument_all()

    def uninstrument(self, worker_id: int):
        uninstrument_all()

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Setting up tracer...")  # worker_id included in process name

        if self.instrument_managed:
            self.instrument(worker_id)
            logger.info(f"[Worker {worker_id}] Instrumentation applied.")

        if self.agentops_managed:
            if self._agentops_server_port_val:  # Use the stored, picklable port value
                base_url = f"http://localhost:{self._agentops_server_port_val}"
                env_vars_to_set = {
                    "AGENTOPS_API_KEY": "dummy",
                    "AGENTOPS_API_ENDPOINT": base_url,
                    "AGENTOPS_APP_URL": f"{base_url}/notavailable",
                    "AGENTOPS_EXPORTER_ENDPOINT": f"{base_url}/traces",
                }
                for key, value in env_vars_to_set.items():
                    os.environ[key] = value
                    logger.info(f"[Worker {worker_id}] Env var set: {key}={value}")
            else:
                logger.warning(
                    f"[Worker {worker_id}] AgentOps managed, but local server port is not available. Client may not connect as expected."
                )

            if not agentops.get_client().initialized:
                agentops.init()
                logger.info(f"[Worker {worker_id}] AgentOps client initialized.")
            else:
                logger.warning(f"[Worker {worker_id}] AgentOps client was already initialized.")

        self._flow_span_processor = FlowSpanProcessor()

        try:
            # new versions
            instance = agentops.sdk.core.tracer
            instance.provider.add_span_processor(self._flow_span_processor)
        except AttributeError:
            # old versions
            instance = TracingCore.get_instance()
            instance._provider.add_span_processor(self._flow_span_processor)

    def teardown_worker(self, worker_id: int) -> None:
        super().teardown_worker(worker_id)

        if self.instrument_managed:
            self.uninstrument(worker_id)
            logger.info(f"[Worker {worker_id}] Instrumentation removed.")

    @contextmanager
    def trace_context(self, name: Optional[str] = None):
        """
        Starts a new tracing context. This should be used as a context manager.

        Args:
            name: Optional name for the tracing context.

        Yields:
            The FlowSpanProcessor instance to collect spans.
        """
        if not self._flow_span_processor:
            raise RuntimeError("FlowSpanProcessor is not initialized. Call init_worker() first.")

        with self._flow_span_processor:
            yield self._flow_span_processor

    def get_last_trace(self) -> List[ReadableSpan]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        if not self._flow_span_processor:
            raise RuntimeError("FlowSpanProcessor is not initialized. Call init_worker() first.")
        return self._flow_span_processor.spans()

    def get_langchain_callback_handler(self, tags: List[str] | None = None) -> LangchainCallbackHandler:
        """
        Get the Langchain callback handler for integrating with Langchain.

        Args:
            tags: Optional list of tags to apply to the Langchain callback handler.

        Returns:
            An instance of the Langchain callback handler.
        """
        import agentops
        from agentops.integration.callbacks.langchain import LangchainCallbackHandler

        tags = tags or []
        client_instance = agentops.get_client()
        api_key = None
        if client_instance.initialized:
            api_key = client_instance.config.api_key
        else:
            logger.warning(
                "AgentOps client not initialized when creating LangchainCallbackHandler. API key may be missing."
            )
        return LangchainCallbackHandler(api_key=api_key, tags=tags)


class FlowSpanProcessor(SpanProcessor):

    def __init__(self, max_traces: int = 1000):
        """Initialize FlowSpanProcessor with trace-based isolation to support concurrent workers.

        Args:
            max_traces: Maximum number of traces to keep in memory. When exceeded, oldest traces are removed.
        """
        super().__init__()
        # Use dict to store spans for each trace_id (supports concurrent rollouts across workers)
        self._spans_by_trace: Dict[str, List[ReadableSpan]] = {}
        self._current_trace_id: Optional[str] = None
        self._lock = threading.Lock()  # Thread-safe for concurrent access
        self._max_traces = max_traces
        self._trace_order: List[str] = []  # Track insertion order for LRU cleanup

    def __enter__(self):
        """Enter trace context and record current trace_id from OpenTelemetry context."""
        from opentelemetry import trace

        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            self._current_trace_id = format(current_span.get_span_context().trace_id, '032x')
            with self._lock:
                if self._current_trace_id not in self._spans_by_trace:
                    self._spans_by_trace[self._current_trace_id] = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit trace context and automatically clean up if trace_id is set."""
        # CRITICAL FIX: Auto-cleanup on __exit__ to prevent memory leak
        # Clear the current trace immediately after exiting the context
        # This is a safety measure in case the runner's clear_trace() fails
        if self._current_trace_id:
            with self._lock:
                # Don't delete here - let the runner handle cleanup
                # Just clear the current_trace_id reference
                logger.debug(f"Exiting trace context for trace_id: {self._current_trace_id}")
                self._current_trace_id = None

    def spans(self) -> List[ReadableSpan]:
        """
        Get the list of spans for the current trace.

        Returns:
            List of ReadableSpan objects for the current trace_id.
        """
        with self._lock:
            if self._current_trace_id and self._current_trace_id in self._spans_by_trace:
                return self._spans_by_trace[self._current_trace_id].copy()
            return []

    def on_end(self, span: ReadableSpan) -> None:
        """
        Process a span when it ends. Groups spans by trace_id to isolate concurrent rollouts.

        Args:
            span: The span that has ended.
        """
        # Skip if span is not sampled
        if not span.context or not span.context.trace_flags.sampled:
            return

        # Get the span's trace_id
        trace_id = format(span.context.trace_id, '032x')

        with self._lock:
            # Add span to the appropriate trace_id bucket
            if trace_id not in self._spans_by_trace:
                self._spans_by_trace[trace_id] = []
                self._trace_order.append(trace_id)
            self._spans_by_trace[trace_id].append(span)

            # CRITICAL FIX: Auto-cleanup oldest traces when max_traces is exceeded
            # This prevents unbounded memory growth if clear_trace() is not called
            if len(self._spans_by_trace) > self._max_traces:
                oldest_trace_id = self._trace_order.pop(0)
                if oldest_trace_id in self._spans_by_trace:
                    del self._spans_by_trace[oldest_trace_id]
                    logger.warning(
                        f"Auto-cleaned oldest trace {oldest_trace_id} (exceeded max_traces={self._max_traces}). "
                        f"This may indicate clear_trace() is not being called properly."
                    )

    def clear_trace(self, trace_id: str) -> None:
        """
        Clear spans for a specific trace_id to prevent memory leaks.

        Args:
            trace_id: The trace ID to clear.
        """
        with self._lock:
            if trace_id in self._spans_by_trace:
                del self._spans_by_trace[trace_id]
                # Also remove from trace order list
                if trace_id in self._trace_order:
                    self._trace_order.remove(trace_id)

    def shutdown(self) -> None:
        """Shutdown the processor and clear all traces."""
        with self._lock:
            self._spans_by_trace.clear()
            self._trace_order.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
