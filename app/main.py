import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import time
from typing import Optional

from app.services.caption import CaptionService
from app.services.feature_extraction import FeatureExtractionService
from app.services.metadata_generation import MetadataGenerationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NFT Image Metadata Extraction Service")

# Initialize services as None
caption_service: Optional[CaptionService] = None
feature_service: Optional[FeatureExtractionService] = None
llm_service: Optional[MetadataGenerationService] = None

@app.on_event("startup")
async def startup_event():
    global caption_service, feature_service, llm_service
    try:
        logger.info("Initializing services...")
        caption_service = CaptionService()
        feature_service = FeatureExtractionService()
        llm_service = MetadataGenerationService()
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # In a real production app, you might want to exit here
        # or handle it so /health reflects the failure

@app.get("/health")
async def health():
    if all([caption_service, feature_service, llm_service]):
        return {"status": "ok"}
    return {"status": "error", "detail": "Services not fully initialized"}

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    if not all([caption_service, feature_service, llm_service]):
        raise HTTPException(status_code=503, detail="Service is still initializing or failed to start")
        
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    try:
        # Load image
        img_content = await image.read()
        pil_image = Image.open(io.BytesIO(img_content)).convert("RGB")
        
        # Step 1: BLIP Captioning
        caption = caption_service.generate_caption(pil_image)
        logger.info(f"Generated caption: {caption}")
        
        # Step 2: CLIP Feature Extraction
        features = feature_service.extract_features(pil_image)
        logger.info(f"Extracted features: {features}")
        
        # Step 3: LLM Metadata Generation
        metadata = llm_service.generate_metadata(caption, features)
        
        latency = time.time() - start_time
        logger.info(f"Analysis completed in {latency:.2f}s")
        
        return metadata
    
    except Exception as e:
        logger.exception("Error during analysis")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
