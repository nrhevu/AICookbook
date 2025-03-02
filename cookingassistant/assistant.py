import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from opentelemetry import trace

from cookingassistant.data.item import Ingredient, Recipe
from cookingassistant.data.suggestor import RecipeSuggestor
from cookingassistant.model.detector import ImageRecognitionModel
from cookingassistant.model.llm import InstructionGenerator

from PIL.Image import Image

class CookingAssistant:
    """Main class that orchestrates the entire workflow"""
    
    def __init__(self, 
                 image_model: ImageRecognitionModel,
                 recipe_processor: RecipeSuggestor,
                 instruction_generator: InstructionGenerator):
        self.image_model = image_model
        self.recipe_processor = recipe_processor
        self.instruction_generator = instruction_generator
    
    def process_request(self, images: List[str] | List[Image], user_query: str) -> Dict[str, Any]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("CookingAssistant.process_request"):
            """Process a user request with images and text query"""
            # read image to Pillow Image if path is provided
            if isinstance(images[0], str):
                # load image
                images = [Image.open(img_path) for img_path in images]
            
            # 1. Recognize ingredients from images
            ingredient_names = self.image_model.predict(images)
            ingredients = [Ingredient(name) for name in ingredient_names]
            
            # 2. Find matching recipes
            matching_recipes = self.recipe_processor.find_matching_recipes(ingredients, user_query)
            
            # 3. Rank recipes by relevance
            ranked_recipes = self.recipe_processor.rank_recipes(matching_recipes, user_query)
            
            if not ranked_recipes:
                return {"status": "no_recipes_found", "message": "No matching recipes found"}
            
            # 4. Generate cooking instructions for the top recipe
            top_recipe = ranked_recipes[0]
            instructions = self.instruction_generator.generate_instructions(
                top_recipe, ingredients, user_query
            )
            
            return {
                "status": "success",
                "recipe": top_recipe,
                "detailed_instructions": instructions,
                "alternative_recipes": ranked_recipes[1:5] if len(ranked_recipes) > 1 else []
            }