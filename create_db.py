import pandas as pd
from tqdm import tqdm

from cookingassistant.data.item import Ingredient, Recipe
from cookingassistant.database import VectorRecipeDatabase
from pymilvus import MilvusException

# Connect to the database
vectordb = VectorRecipeDatabase()
vectordb.connect("localhost:19530")

# Load the dataset
df = pd.read_json("./data/recipes_with_nutritional_info.json")
df["ingredients"] = df["ingredients"].apply(lambda x: [i["text"] for i in x])
df["instructions"] = df["instructions"].apply(lambda x: "\n".join([i["text"] for i in x]))

# Add all ingredients to the database
ingredients_name = set()
for ingredient in df["ingredients"].values:
    for i in ingredient:
        ingredients_name.add(i)

ingredients = {}
for i in ingredients_name:
    ingredients[i] = Ingredient(i, is_common=False)

# Add all recipes to the database
for i, d in tqdm(df.iterrows(), total=len(df)):
    try:
        recipe = Recipe(id=d["id"], name=d["title"], instructions=d["instructions"], ingredients=[ingredients[i] for i in d["ingredients"]])
        vectordb.add_recipe(recipe)
    except MilvusException as e:
        print("Error", e)