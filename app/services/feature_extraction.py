import torch
import open_clip
from PIL import Image
import logging

class FeatureExtractionService:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading OpenCLIP model {model_name}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device="cpu"
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.logger.info("OpenCLIP model loaded on CPU.")
        
        # Predefined categories for zero-shot extraction
        self.styles = ["digital art", "oil painting", "watercolor", "sketch", "3d render", "pixel art", "cyberpunk", "vaporwave", "minimalist", "abstract", "realistic", "anime", "pop art"]
        self.colors = ["vibrant", "monochrome", "pastel", "dark", "neon", "warm colors", "cool colors", "gold", "silver"]
        self.moods = ["happy", "gloomy", "energetic", "peaceful", "aggressive", "mysterious", "futuristic", "nostalgic"]

    def extract_features(self, image: Image.Image) -> dict:
        try:
            image_input = self.preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                style = self._get_best_match(image_features, self.styles)
                color = self._get_best_match(image_features, self.colors)
                mood = self._get_best_match(image_features, self.moods)
                
                return {
                    "style_candidate": style,
                    "color_candidate": color,
                    "mood_candidate": mood
                }
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise e

    def _get_best_match(self, image_features, candidates):
        text_tokens = self.tokenizer(candidates)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            return candidates[indices[0]]
