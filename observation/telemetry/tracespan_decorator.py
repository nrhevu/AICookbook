import requests
from functools import wraps
from opentelemetry import trace

class TraceSpan:
    def __init__(self, span_name: str):
        self.span_name = span_name

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(func.__module__)
            with tracer.start_as_current_span(self.span_name):
                return func(*args, **kwargs)
        return wrapper