import unittest
from unittest.mock import MagicMock, patch
import sys

# Mocking heavy libraries
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['open_clip'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()

from PIL import Image
import io
import os

from app.services.caption import CaptionService
from app.services.feature_extraction import FeatureExtractionService
from app.services.metadata_generation import MetadataGenerationService

class TestServicesMock(unittest.TestCase):
    @patch('app.services.caption.BlipProcessor.from_pretrained')
    @patch('app.services.caption.BlipForConditionalGeneration.from_pretrained')
    def test_caption_service(self, mock_model_from_pretrained, mock_processor_from_pretrained):
        # Mock processor and model
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_from_pretrained.return_value = mock_processor
        mock_model_from_pretrained.return_value = mock_model
        
        service = CaptionService(model_path="mock_path")
        
        # Mock BatchEncoding
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        
        mock_model.generate.return_value = [MagicMock()]
        mock_processor.decode.return_value = "A beautiful sunset"
        
        img = Image.new('RGB', (100, 100))
        caption = service.generate_caption(img)
        self.assertEqual(caption, "A beautiful sunset")

    @patch('app.services.feature_extraction.open_clip.create_model_and_transforms')
    @patch('app.services.feature_extraction.open_clip.get_tokenizer')
    def test_feature_extraction_service(self, mock_tokenizer, mock_create):
        # Mock CLIP
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_create.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer.return_value = MagicMock()
        
        service = FeatureExtractionService()
        
        # Mock _get_best_match
        service._get_best_match = MagicMock()
        service._get_best_match.side_effect = ["Cyberpunk", "Neon blue", "Futuristic"]
        
        img = Image.new('RGB', (100, 100))
        features = service.extract_features(img)
        
        self.assertEqual(features['style_candidate'], "Cyberpunk")
        self.assertEqual(features['color_candidate'], "Neon blue")
        self.assertEqual(features['mood_candidate'], "Futuristic")

    @patch('app.services.metadata_generation.Llama')
    def test_metadata_generation_service(self, mock_llama):
        mock_llm_instance = MagicMock()
        mock_llama.return_value = mock_llm_instance
        
        mock_llm_instance.return_value = {
            'choices': [{
                'text': '{"Title": "Cyber Sun", "Description": "Sun in cyberpunk city", "Style": "Cyberpunk", "Color": "Neon", "Mood": "Futuristic", "Tags": "cyber, sun, neon"}'
            }]
        }
        
        service = MetadataGenerationService(model_path="mock_path")
        metadata = service.generate_metadata("A beautiful sunset", {"style_candidate": "Cyberpunk", "color_candidate": "Neon blue", "mood_candidate": "Futuristic"})
        
        self.assertEqual(metadata['title'], "Cyber Sun")
        self.assertEqual(metadata['style'], "Cyberpunk")

if __name__ == '__main__':
    unittest.main()
