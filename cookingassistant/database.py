import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      MilvusClient, connections, utility)
from sentence_transformers import SentenceTransformer

from cookingassistant.data.item import Ingredient, Recipe


class CommonIngredientsRegistry:
    """Registry for common ingredients that should be ignored in recipe matching"""

    def __init__(self):
        self.common_ingredients = self._initialize_common_ingredients()

    def _initialize_common_ingredients(self) -> List[Ingredient]:
        """Initialize the list of common ingredients"""
        from cookingassistant.data.item import Ingredient

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
        return any(
            ing.name.lower() == ingredient_name.lower()
            for ing in self.common_ingredients
        )

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
    def find_recipes_by_ingredients(
        self,
        ingredients: List[Ingredient],
        category: Optional[str] = None,
        top_n: int = 3,
    ) -> List["Recipe"]:
        """Find recipes that match the given ingredients, optionally filtered by category"""
        pass

    @abstractmethod
    def get_recipe_by_id(self, recipe_id: str) -> Optional[Recipe]:
        """Get a specific recipe by ID"""
        pass


class SQLRecipeDatabase(RecipeDatabase):
    pass


class VectorRecipeDatabase(RecipeDatabase):
    """Vector database implementation using Milvus for recipe storage and retrieval"""

    # Milvus collection names
    RECIPE_COLLECTION = "recipes"

    # Vector dimensions - depends on the embedding model used
    VECTOR_DIM = 384

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the vector database with an embedding model"""
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.client = None
        self.connected = False

    def connect(self, connection_string: str) -> None:
        """Connect to the Milvus database"""
        try:
            host, port = connection_string.split(":")

            # Connect to Milvus server
            connections.connect(host=host, port=port)

            # Create Milvus client
            self.client = MilvusClient(uri=f"http://{connection_string}")

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Create collections if they don't exist
            self._initialize_collections()
            
            self.connected = True
            print(f"Successfully connected to Milvus at {connection_string}")

        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            self.connected = False

    def _initialize_collections(self) -> None:
        """Initialize Milvus collections if they don't exist"""
        # Recipe collection
        if not utility.has_collection(self.RECIPE_COLLECTION):
            recipe_fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100
                ),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.VECTOR_DIM
                ),
                FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=10000),
            ]
            recipe_schema = CollectionSchema(fields=recipe_fields)
            recipe_collection = Collection(
                name=self.RECIPE_COLLECTION, schema=recipe_schema
            )

            # Create index for vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
            recipe_collection.create_index(
                field_name="vector", index_params=index_params
            )
            print("Index created successfully")
        
            # Load the collection
            recipe_collection.load() 

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for text"""
        if not self.embedding_model:
            raise ValueError(
                "Embedding model not initialized. Connect to database first."
            )

        # Generate embedding
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def add_recipe(self, recipe: Recipe) -> None:
        """
        Add a recipe to the vector database
        """

        if not self.connected:
            raise ConnectionError("Not connected to Milvus database")

        # Generate embedding for recipe (combine name and ingredients)
        ingredient_names = [ing.name for ing in recipe.ingredients]
        recipe_text = f"{recipe.name} {' '.join(ingredient_names)}"
        recipe_vector = self._generate_embedding(recipe_text)

        # Serialize recipe data
        recipe_data = {
            "id": recipe.id,
            "name": recipe.name,
            "ingredients": [
                {"name": ing.name, "is_common": ing.is_common}
                for ing in recipe.ingredients
            ],
            "instructions": recipe.instructions,
        }

        self.client.insert(
            collection_name=self.RECIPE_COLLECTION,
            data={
                "id": recipe.id,
                "name": recipe.name,
                "vector": recipe_vector,
                "data": json.dumps(recipe_data),
            },
        )

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Recipe]:
        """
        Get a specific recipe by ID
        """
        from cookingassistant.data.item import Ingredient, Recipe

        if not self.connected:
            raise ConnectionError("Not connected to Milvus database")

        collection = Collection(self.RECIPE_COLLECTION)
        if utility.load_state(self.RECIPE_COLLECTION) != "Loaded":
            collection.load()

        # Query Milvus
        results = self.client.query(
            collection_name=self.RECIPE_COLLECTION,
            filter=f'id == "{recipe_id}"',
            output_fields=["id", "name", "data"],
        )

        if not results:
            return None

        # Process result
        recipe_data = json.loads(results[0]["data"])
        ingredients_list = [
            Ingredient(name=ing["name"], is_common=ing["is_common"])
            for ing in recipe_data["ingredients"]
        ]

        recipe = Recipe(
            id=recipe_data["id"],
            name=recipe_data["name"],
            ingredients=ingredients_list,
            instructions=recipe_data["instructions"],
        )

        return recipe

    def find_recipes_by_ingredients(
        self,
        ingredients: List[Ingredient],
        category: Optional[str] = None,
        top_n: int = 3,
    ) -> List[Recipe]:
        """
        Find recipes that match the given ingredients
        """

        if not self.connected:
            raise ConnectionError("Not connected to Milvus database")

        collection = Collection(self.RECIPE_COLLECTION)
        if utility.load_state(self.RECIPE_COLLECTION) != "Loaded":
            collection.load()

        if not ingredients:
            return []

        # Generate a combined query vector from all ingredients
        ingredient_names = [ing.name for ing in ingredients]
        query_text = " ".join(ingredient_names)
        query_vector = self._generate_embedding(query_text)

        # Search for similar recipes
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        # Build filtering expression for category if provided
        expr = None
        if category:
            expr = f'category like "%{category}%"'

        # Perform search
        recipe_search_results = self.client.search(
            collection_name=self.RECIPE_COLLECTION,
            data=[query_vector],
            filter=expr,
            limit=3,
            output_fields=["id", "name", "data"],
            search_params=search_params,
        )

        # Process results
        matching_recipes = []

        if not recipe_search_results:
            print("No matching recipes.")
        else:
            for recipe in recipe_search_results[0]:
                entity = recipe.get("entity", {})
                data_json = entity.get("data", "{}")
                recipe = json.loads(data_json)
                matching_recipes.append(recipe)

        return matching_recipes
