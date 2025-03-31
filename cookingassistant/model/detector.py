import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import PIL
from PIL.Image import Image
from ultralytics import YOLO

from observation.telemetry.tracespan_decorator import TraceSpan


class ImageRecognitionModel(ABC):
    """Abstract class for the AI model that recognizes ingredients"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the trained model from the specified path"""
        pass
    
    @abstractmethod
    def predict(self, images: List[Image]) -> List[str]:
        """Predict ingredients from images and return their names"""
        pass
    
class PyTorchImageRecognitionModel(ImageRecognitionModel):
    """Concrete implementation of a PyTorch-based recognition model"""
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load the PyTorch model"""
        # Implementation would load the model
        return YOLO(model_path)
        
    @TraceSpan("PyTorchImageRecognitionModel.predict")
    def predict(self, images: List[Image]) -> Set[str]:
        
        """Predict ingredients using the PyTorch model"""
        # Implementation would run inference on images
        ingredients = set()
        for i in images:
            # i = PIL.Image.open(i)
            results = self.model(i)

            # Lấy danh sách các vật thể phát hiện được
            detected_objects = []
            for r in results:
                for box in r.boxes:
                    detected_objects.append(
                        {
                            "class": self.model.names[int(box.cls)],  # Tên lớp
                            "confidence": float(box.conf),  # Độ tin cậy
                        }
                    )
            for obj in detected_objects:
                ingredients.add(obj["class"])
        return ingredients

    #Predict with 
    #def predict(self, images: List[Image]) -> List[str]:
    #    """Predict ingredients using the PyTorch model"""
    #    # Implementation would run inference on images
    #    return []

class OnnxImageRecognitionModel(ImageRecognitionModel):
    pass

