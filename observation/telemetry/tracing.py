from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def init_tracer(service_name: str = "cooking-assistant", collector_endpoint: str = "http://localhost:14268/api/traces"):
    """
    Initialize the OpenTelemetry tracer with a Jaeger HTTP exporter.
    """
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    jaeger_exporter = JaegerExporter(
        collector_endpoint=collector_endpoint,
    )
    span_processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(span_processor)