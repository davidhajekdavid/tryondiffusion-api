from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
import traceback
from typing import Optional
import logging
import os
from PIL import Image, ImageDraw
import numpy as np

# Environment configuration
PORT = int(os.getenv("PORT", 8001))
ENV = os.getenv("ENVIRONMENT", "production")
DEBUG = ENV == "development"

# Configure logging
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TryOnDiffusion API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """Load and convert uploaded image to PIL Image"""
    contents = upload_file.file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def create_mock_tryon_result(person_img: Image.Image, garment_img: Image.Image) -> Image.Image:
    """Create a mock try-on result by blending images"""
    # Resize images to same size
    size = (256, 256)
    person_resized = person_img.resize(size, Image.Resampling.LANCZOS)
    garment_resized = garment_img.resize(size, Image.Resampling.LANCZOS)
    
    # Create a simple blend effect as a placeholder
    # In a real implementation, this would be the AI model output
    result = Image.blend(person_resized, garment_resized, 0.3)
    
    # Add some overlay effect to make it look more like a try-on
    draw = ImageDraw.Draw(result)
    draw.rectangle([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], 
                   outline=(255, 0, 0), width=2)
    
    return result

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "TryOnDiffusion API is running!"}

@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy",
        "model_status": "ready",
        "device": "cpu",
        "message": "TryOnDiffusion API is running! (Mock mode for deployment testing)"
    }

@app.post("/tryon")
async def virtual_tryon(
    person_image: UploadFile = File(..., description="Person/model image"),
    garment_image: UploadFile = File(..., description="Clothing item image"),
    clothing_agnostic_image: Optional[UploadFile] = File(None, description="Clothing-agnostic person image (optional)"),
    guidance_scale: float = Form(2.0, description="Guidance scale for generation"),
    num_steps: int = Form(20, description="Number of inference steps")
):
    """
    Virtual try-on endpoint (Mock implementation for testing deployment)
    
    - **person_image**: Image of the person/model
    - **garment_image**: Image of the clothing item to try on
    - **clothing_agnostic_image**: Optional clothing-agnostic person image
    - **guidance_scale**: Controls how closely the model follows the input (higher = more faithful)
    - **num_steps**: Number of denoising steps (higher = better quality but slower)
    """
    
    try:
        logger.info("Processing virtual try-on request (mock mode)...")
        
        # Load images
        person_img = load_image_from_upload(person_image)
        garment_img = load_image_from_upload(garment_image)
        
        logger.info(f"Loaded images - Person: {person_img.size}, Garment: {garment_img.size}")
        
        # Create mock result
        result_image = create_mock_tryon_result(person_img, garment_img)
        
        # Convert to base64 for response
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("Virtual try-on completed successfully! (Mock mode)")
        
        return {
            "success": True,
            "result_image": f"data:image/png;base64,{image_base64}",
            "message": "Virtual try-on completed successfully! (Mock mode - real AI model coming soon)",
            "mock_mode": True
        }
        
    except Exception as e:
        logger.error(f"Error in virtual try-on: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")

@app.post("/tryon_simple")
async def virtual_tryon_simple(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    """
    Simplified virtual try-on endpoint with default parameters
    """
    return await virtual_tryon(
        person_image=person_image,
        garment_image=garment_image,
        clothing_agnostic_image=None,
        guidance_scale=2.0,
        num_steps=20
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT) 