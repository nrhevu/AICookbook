# Load configuration
from cookingassistant.engine import AppConfig, AppFactory
from observation.telemetry.tracing import init_tracer

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Initialize tracing with Jaeger
init_tracer(service_name="cooking-assistant", collector_endpoint="http://localhost:14268/api/traces")


# Create and instrument FastAPI app
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

config = AppConfig(
    model_path="/path/to/model",
    db_connection_string="connection_string",
    openai_api_key="API_KEY",
    use_vector_db=True,
    model_type="pytorch"
)

# Create cooking assistant
assistant = AppFactory.create_cooking_assistant(config)

@app.get("/test")
def test_endpoint():
    """
    A simple test endpoint that returns a greeting message.
    """
    return {"message": "Hello, I am your cooking AI. How can I help you today?"}

@app.get("/")
def suggest(images, user_query):
    result = assistant.process_request(
        images=images,
        user_query=user_query
    )
    return {"Recipe": result['recipe'].name, "Instructions": result['detailed_instructions']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)