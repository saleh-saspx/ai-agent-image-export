import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

class CaptionService:
    def __init__(self, model_path: str = "app/models/blip"):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading BLIP model from {model_path}...")
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        self.device = "cpu"
        self.model.to(self.device)
        self.logger.info("BLIP model loaded on CPU.")

    def generate_caption(self, image: Image.Image) -> str:
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            raise e
