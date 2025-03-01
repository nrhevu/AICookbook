# Load configuration
from cookingassistant import AppConfig, AppFactory
from fastapi import FastAPI

app = FastAPI()

config = AppConfig(
    model_path="/path/to/model",
    db_connection_string="connection_string",
    openai_api_key="API_KEY",
    use_vector_db=True,
    model_type="pytorch"
)

# Create cooking assistant
assistant = AppFactory.create_cooking_assistant(config)

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