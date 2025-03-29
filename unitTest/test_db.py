# test_milvus.py
import unittest
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from cookingassistant.database import VectorRecipeDatabase
from cookingassistant.data.item import Ingredient

class TestMilvus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a connection to Milvus before running the tests."""
        connections.connect(host="localhost", port="19530")
    
    def setUp(self):
        self.vectorDB = VectorRecipeDatabase()
        self.vectordb.connect("localhost:19530")

    @classmethod
    def tearDownClass(cls):
        """Disconnect from Milvus after tests are done."""
        connections.disconnect()

    def test_create_collection(self):
        """Test creating a collection in Milvus."""
        collection_name = "test_collection"
        fields = [
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        ]
        schema = CollectionSchema(fields, description="Test collection for Milvus")
        collection = Collection(name=collection_name, schema=schema)
        
        self.assertTrue(collection.is_exist, "Collection was not created successfully!")
    
    def test_insert_vectors(self):
        """Test inserting vectors into the collection."""
        collection_name = "test_collection"
        collection = Collection(collection_name)

        vectors = [[0.1 * i for i in range(128)] for _ in range(10)]

        ids = collection.insert([vectors])

        # Assert the number of inserted vectors
        self.assertEqual(collection.num_entities, 10, "Number of inserted vectors is not correct.")

    def test_search_vectors(self):
        """Test searching vectors in Milvus."""
        collection_name = "test_collection"
        collection = Collection(collection_name)

        # Prepare a query vector
        query_vector = [[0.1 * i for i in range(128)]]
        
        # Search for the nearest vectors
        results = collection.search(query_vector, "vector", param={"nprobe": 10}, limit=5)

        self.assertGreater(len(results), 0, "Search did not return any results.")
        self.assertGreater(len(results[0]), 0, "Search did not return the expected number of results.")

    def test_drop_collection(self):
        """Test dropping the collection."""
        collection_name = "test_collection"
        collection = Collection(collection_name)

        # Drop the collection
        collection.drop()

        self.assertFalse(collection.is_exist, "Collection wasn't dropped!")
    
    def test_existing_recipe_collection(self):
        collection_name = "recipes"
        collection = Collection(collection_name)
        
        self.assertTrue(collection.is_exist, "Collection was not created successfully!")


    def test_search_recipe_collection_by_ingredients(self):
        ingredients = [Ingredient('tomato', is_common=False), Ingredient('egg', is_common=False)]
        matched_recipes = self.vectorDB.find_recipes_by_ingredients(ingredients)
        bool_matched_ingredients = False
        for recipes in matched_recipes:
            all_ingredient = " ".join(recipes.ingredients)
            if "egg" in bool_matched_ingredients or "tomato" in bool_matched_ingredients:
                bool_matched_ingredients = True
                break
        
        self.assertTrue(bool_matched_ingredients, "Failed to find suitable dishes")


    def test_get_recipe_by_id(self):
        pass
    
    def test_insert_recipe(self):
        pass
    def test_delete_recipe(self):
        pass


if __name__ == "__main__":
    unittest.main()
