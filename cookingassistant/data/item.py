from dataclasses import dataclass
from typing import List


@dataclass
class Ingredient:
    """Class representing a food ingredient"""
    name: str
    is_common: bool = False
    
@dataclass
class Recipe:
    """Class representing a cooking recipe"""
    id: str
    name: str
    ingredients: List[Ingredient]
    instructions: str
    # category: str
    # preparation_time: int  # in minutes
    # difficulty_level: str  # "easy", "medium", "hard"