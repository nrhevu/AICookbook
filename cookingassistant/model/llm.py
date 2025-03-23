import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from cookingassistant.data.item import Ingredient, Recipe
from observation.telemetry.tracespan_decorator import TraceSpan


class LLMClient(ABC):
    """Abstract class for LLM API client"""

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text using the LLM"""
        pass

@TraceSpan("OpenAIClient.generate_text")
class OpenAIClient(LLMClient):
    """Implementation for OpenAI's API"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

    def generate_text(self, prompt: str) -> str:
        """Generate text using OpenAI's API"""
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        prompt = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False  # Disable streaming
        }
        response = requests.post(url, headers=headers, json=prompt)
        return response.text

class BaseInstructionGenerator():
    @abstractmethod
    def generate_instructions(self, recipe: Recipe, available_ingredients: List[Ingredient], user_query: str) -> str:
        pass

class InstructionGeneratorByLLM(BaseInstructionGenerator):
    """Class that generates cooking instructions using LLM"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def create_prompt(self, recipe: Recipe, available_ingredients: List[Ingredient], user_query: str) -> str:
        """Create a prompt for the LLM based on recipe and user query"""
        ingredient_list = ", ".join([ing.name for ing in available_ingredients])
        prompt = (
            f"You are a professional chef assistant. The user wants to cook '{recipe.name}'.\n"
            f"The recipe includes these ingredients: {', '.join([ing.name for ing in recipe.ingredients])}.\n"
            f"The user currently has these ingredients: {ingredient_list}.\n"
            f"The user asks: {user_query}.\n"
            f"Provide step-by-step cooking instructions considering the available ingredients."
        )
        return prompt

    def generate_instructions(self, recipe: Recipe, available_ingredients: List[Ingredient], user_query: str) -> str:
        """Generate detailed cooking instructions using LLM"""
        if self.llm_client.api_key == '':
            raise Exception("No API Key provided")
        prompt = self.create_prompt(recipe, available_ingredients, user_query)
        return self.llm_client.generate_text(prompt)

class InstructionGeneratorByTemplate(BaseInstructionGenerator):
    def generate_instructions(self, recipe: Recipe, available_ingredients: List[Ingredient], user_query: str) -> str:
        final_result = ""
        for i, r in enumerate(recipe):
            final_result += f"{i+1}.\n"
            final_result += f"""Name: {r["name"]}\n"""
            final_result += f"""Ingredients: {', '.join([ing["name"] for ing in r["ingredients"]])}\n"""
            instruction = "\n    -".join(r["instructions"].split("\n"))
            final_result += f"""Instructions:\n    -{instruction}\n\n"""

        return str(final_result)
    