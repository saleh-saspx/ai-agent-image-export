import logging
import json
from llama_cpp import Llama

class MetadataGenerationService:
    def __init__(self, model_path: str = "app/models/llm/phi3-mini-instruct-q4.gguf"):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading LLM from {model_path}...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=8, # Optimized for 8 cores as requested
                verbose=False
            )
            self.logger.info("LLM loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load LLM: {e}")
            raise e

    def generate_metadata(self, caption: str, features: dict) -> dict:
        prompt = f"""<|system|>
You are an expert NFT curator. Convert the following image analysis into NFT metadata.
Output MUST be a valid JSON object with the following fields: Title, Description, Style, Color, Mood, Tags.
Tags should be a comma-separated string of 3-5 keywords.
Use single key-value fields.
<|user|>
Image Analysis:
Caption: {caption}
Detected Style: {features.get('style_candidate')}
Detected Color: {features.get('color_candidate')}
Detected Mood: {features.get('mood_candidate')}

Convert the following image analysis into NFT metadata with fields: Title, Description, Style, Color, Mood, Tags
<|assistant|>
"""
        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                stop=["<|end|>", "\n\n"],
                echo=False,
                temperature=0.7
            )
            
            text_output = response['choices'][0]['text'].strip()
            self.logger.debug(f"LLM raw output: {text_output}")
            
            # Attempt to parse JSON from the output
            # In case the model adds preamble or postamble
            start_idx = text_output.find('{')
            end_idx = text_output.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = text_output[start_idx:end_idx]
                metadata = json.loads(json_str)
                # Ensure keys are lowercase to match requirement
                return {k.lower(): v for k, v in metadata.items()}
            else:
                raise ValueError("Could not find JSON in LLM response")
                
        except Exception as e:
            self.logger.error(f"Error generating metadata: {e}")
            # Fallback metadata if LLM fails
            return {
                "title": caption[:50],
                "description": caption,
                "style": features.get('style_candidate'),
                "color": features.get('color_candidate'),
                "mood": features.get('mood_candidate'),
                "tags": f"{features.get('style_candidate')}, {features.get('mood_candidate')}"
            }
