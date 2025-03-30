import unittest
from cookingassistant.model.detector import PyTorchImageRecognitionModel
from cookingassistant.assistant import CookingAssistant
from cookingassistant.data.item import Ingredient, Recipe
from cookingassistant.data.suggestor import RecipeSuggestor
from cookingassistant.database import CommonIngredientsRegistry, VectorRecipeDatabase
from cookingassistant.model.llm import InstructionGeneratorByTemplate
import requests
from PIL import Image
from io import BytesIO


class TestBackend(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = PyTorchImageRecognitionModel('./models/best.pt')
        cls.common_ingredients = CommonIngredientsRegistry()
        cls.vectordb = VectorRecipeDatabase()
        cls.vectordb.connect("localhost:19530")
        cls.recipe_processor = RecipeSuggestor(cls.vectordb, cls.common_ingredients)
        cls.instruction_generator = InstructionGeneratorByTemplate()
        cls.cooking_assistant = CookingAssistant(cls.model, cls.recipe_processor, cls.instruction_generator)
    
    def test_model_predict(cls):
        # image having some carrots
        url = 'https://theseedcompany.ca/cdn/shop/files/crop_CARR1923_Carrot___Sweetness_Pelleted_Long.png?v=1720113309&width=1024'
        response = requests.get(url)
        image = None
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
        else:
            raise Exception("Failed to download image")
    
        ingredient_names = cls.model.predict([image])
        cls.assertIsInstance(ingredient_names, set, "Result is not a set")

        cls.assertTrue(all(isinstance(item, str) for item in ingredient_names), "Not all elements are strings")

    def test_embedding_model(cls):
        vector = cls.vectordb._generate_embedding("tomato")

        cls.assertEqual(len(vector), 384, "Not match required vector size")
        cls.assertTrue(all(isinstance(item, float) for item in vector), "Not all elements are float")


    def test_find_matching_food(cls):
        ingredients = [Ingredient(name) for name in ["potato", "chicken"]]
        matching_recipes = cls.recipe_processor.find_matching_recipes(ingredients, "")
        full_ingre = ""
        for recipes in matching_recipes:
            for ingre in recipes['ingredients']:
                full_ingre += ingre['name']
        
        full_ingre = full_ingre.lower()
        print("Full ingredients: ", full_ingre)
        boolMatching = False
        if "potato" in full_ingre or "chicken" in full_ingre:
            boolMatching = True

        cls.assertIsInstance(matching_recipes, list, "Is not a list")
        cls.assertTrue(boolMatching, "Can not find matching food")


    def test_find_reranking(cls):
        ingredients = [Ingredient(name) for name in ["egg", "beef"]]
        matching_recipes = cls.recipe_processor.find_matching_recipes(ingredients, "")
        ranked_recipes = cls.recipe_processor.rank_recipes(matching_recipes, "")

        cls.assertIsInstance(ranked_recipes, list, "Is not a list")
        cls.assertEqual(len(ranked_recipes), len(matching_recipes), "Not match required vector size")
        # cls.assertTrue(all(isinstance(item, Recipe) for item in ranked_recipes), "Not all elements are recipe")
