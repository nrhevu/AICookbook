import unittest
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from cookingassistant.database import VectorRecipeDatabase
from cookingassistant.data.item import Ingredient, Recipe

class TestMilvus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a connection to Milvus before running the tests."""
        connections.connect(host="localhost", port="19530")
        cls.vectorDB = VectorRecipeDatabase()
        cls.vectorDB.connect("localhost:19530")
    
    def setUp(self):
        super().setUp()
        self.vectorDB = VectorRecipeDatabase()
        self.vectorDB.connect("localhost:19530")    

    @classmethod
    def tearDownClass(cls):
        """Disconnect from Milvus after tests are done."""
        connections.disconnect(alias="default")

    def test_create_collection(cls):
        """Test creating a collection in Milvus."""
        collection_name = "test_collection"

        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.drop()

        fields = [
            FieldSchema(name="vector_test", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, enable_dynamic_field=True)
        ]
        schema = CollectionSchema(fields, description="Test collection for Milvus")
        collection = Collection(name=collection_name, schema=schema)
        index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
        collection.create_index(
                field_name="vector_test", index_params=index_params
            )

        cls.assertTrue(utility.has_collection(collection_name), "Collection was not created successfully!")
    
    def test_insert_vectors(self):
        """Test inserting vectors into the collection."""
        collection_name = "test_collection_2"
        if utility.has_collection(collection_name):
            collection = Collection(name=collection_name)
            collection.drop()

        fields = [
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
        ]

        schema = CollectionSchema(fields, description="Test collection for Milvus")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }

        collection.create_index(
            field_name="vector", index_params=index_params
        )

        vectors = [[0.1 * i for i in range(128)] for _ in range(10)]
        for i in range(len(vectors)):
            res = self.vectorDB.client.insert(
                collection_name=collection_name,
                data = {"id": i, 
                        "vector": vectors[i]
                        }
                )
            
        collection.flush()

        collection = Collection(name=collection_name)
        if utility.load_state(collection_name) != "Loaded":
            print('load collection')
            collection.load()

        self.assertEqual(collection.num_entities, 10, "Number of inserted vectors is not correct.")

    def test_search_vectors(cls):
        """Test searching vectors in Milvus."""
        collection_name = "test_collection_2"
        collection = Collection(collection_name)

        # Prepare a query vector
        query_vector = [[0.1 * i for i in range(128)]]
        
        # Search for the nearest vectors
        results = collection.search(query_vector, "vector", param={"nprobe": 10}, limit=5)

        cls.assertGreater(len(results), 0, "Search did not return any results.")
        cls.assertGreater(len(results[0]), 0, "Search did not return the expected number of results.")

    def test_drop_collection(cls):
        """Test dropping the collection."""
        collection_name = "test_collection"
        collection = Collection(collection_name)

        # Drop the collection
        collection.drop()

        cls.assertFalse(utility.has_collection(collection_name), "Collection wasn't dropped!")
    
    def test_existing_recipe_collection(cls):
        collection_name = "recipes"
        
        cls.assertTrue(utility.has_collection(collection_name), "Collection was not created successfully!")

    def test_search_recipe_collection_by_ingredients(cls):
        ingredients = [Ingredient('tomato', is_common=False), Ingredient('egg', is_common=False)]
        matched_recipes = cls.vectorDB.find_recipes_by_ingredients(ingredients)
        bool_matched_ingredients = False
        for recipes in matched_recipes:
            all_ingredient = ""
            for ingr in recipes['ingredients']:
                all_ingredient += ingr['name']
            
            all_ingredient = all_ingredient.lower()
            if "egg" in all_ingredient or "tomato" in all_ingredient:
                bool_matched_ingredients = True
                break
        
        cls.assertTrue(bool_matched_ingredients, "Failed to find suitable dishes")


    def test_insert_recipe(self):
        ingredient_dict = {0: Ingredient("test1", is_common=False),1: Ingredient("test2", is_common=False)}
        recipe = Recipe(id="test_id", name="test_name", instructions="test", ingredients=[ingredient_dict[i] for i in ingredient_dict])
        collection = Collection('recipes')
        self.vectorDB.add_recipe(recipe)
        collection.flush()
        recipe_test = self.vectorDB.get_recipe_by_id("test_id")
        self.assertEqual(recipe_test.id, "test_id", "Failed to insert")
        self.vectorDB.client.delete(ids=['test_id'], collection_name="recipes")
         
    def test_delete_recipe(self):
        ingredient_dict = {0: Ingredient("test1", is_common=False),1: Ingredient("test2", is_common=False)}
        recipe = Recipe(id="test_id_1", name="test_name", instructions="test", ingredients=[ingredient_dict[i] for i in ingredient_dict])
        self.vectorDB.add_recipe(recipe)
        self.vectorDB.client.delete(ids=['test_id_1'], collection_name="recipes")
        recipe_test = self.vectorDB.get_recipe_by_id("test_id_1")

        self.assertIsNone(recipe_test, "Failed to delete")


if __name__ == "__main__":
    unittest.main()
