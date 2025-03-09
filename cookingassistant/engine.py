from cookingassistant.assistant import CookingAssistant
from cookingassistant.data.suggestor  import RecipeSuggestor
from cookingassistant.database import CommonIngredientsRegistry, SQLRecipeDatabase, VectorRecipeDatabase
from cookingassistant.model.detector import PyTorchImageRecognitionModel
from cookingassistant.model.llm import InstructionGenerator, OpenAIClient


class AppConfig:
    """Configuration class for the application"""
    
    def __init__(self, 
                 model_path: str,
                 db_connection_string: str,
                 openai_api_key: str,
                 use_vector_db: bool = False,
                 model_type: str = "pytorch"):
        self.model_path = model_path
        self.db_connection_string = db_connection_string
        self.openai_api_key = openai_api_key
        self.use_vector_db = use_vector_db
        self.model_type = model_type


class AppFactory:
    """Factory class to create the application components"""
    
    @staticmethod
    def create_cooking_assistant(config: AppConfig) -> CookingAssistant:
        """Create and configure the CookingAssistant instance"""
        # Create image recognition model
        if config.model_type.lower() == "pytorch":
            image_model = PyTorchImageRecognitionModel(config.model_path)
        else:
            pass
        
        # Create recipe database
        if config.use_vector_db:
            recipe_db = VectorRecipeDatabase(config.db_connection_string)
        else:
            recipe_db = SQLRecipeDatabase(config.db_connection_string)
        
        # Create common ingredients registry
        common_ingredients = CommonIngredientsRegistry()
        
        # Create recipe processor
        recipe_processor = RecipeSuggestor(recipe_db, common_ingredients)
        
        # Create LLM client and instruction generator
        llm_client = OpenAIClient(config.openai_api_key)
        instruction_generator = InstructionGenerator(llm_client)
        
        # Create and return the cooking assistant
        return CookingAssistant(image_model, recipe_processor, instruction_generator)

