import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cookingassistant.data import Ingredient, Recipe


class CommonIngredientsRegistry:
    """Registry for common ingredients that should be ignored in recipe matching"""
    
    def __init__(self):
        self.common_ingredients = self._initialize_common_ingredients()
    
    def _initialize_common_ingredients(self) -> List[Ingredient]:
        """Initialize the list of common ingredients"""
        return [
            Ingredient("salt", True),
            Ingredient("pepper", True),
            Ingredient("onion", True),
            Ingredient("garlic", True),
            Ingredient("tomato", True),
            Ingredient("oil", True),
            # Add more common ingredients
        ]
    
    def is_common(self, ingredient_name: str) -> bool:
        """Check if an ingredient is in the common ingredients list"""
        return any(ing.name.lower() == ingredient_name.lower() for ing in self.common_ingredients)
    
    def get_all_common_ingredients(self) -> List[Ingredient]:
        """Get all common ingredients"""
        return self.common_ingredients
    
class RecipeDatabase(ABC):
    """Abstract class for recipe database access"""
    
    @abstractmethod
    def connect(self, connection_string: str) -> None:
        """Connect to the database"""
        pass
    
    @abstractmethod
    def find_recipes_by_ingredients(self, ingredients: List[Ingredient], category: Optional[str] = None) -> List[Recipe]:
        """Find recipes that match the given ingredients, optionally filtered by category"""
        pass
    
    @abstractmethod
    def get_recipe_by_id(self, recipe_id: str) -> Optional[Recipe]:
        """Get a specific recipe by ID"""
        pass

class SQLRecipeDatabase(RecipeDatabase):
    pass

class VectorRecipeDatabase(RecipeDatabase):
    pass

