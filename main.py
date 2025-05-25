#!/usr/bin/env python3
"""
Standalone TryOnDiffusion API for Render.com deployment
This file has no external dependencies on the complex TryOnDiffusion package
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import traceback
from typing import Optional
import logging
import os
from PIL import Image, ImageDraw, ImageFilter
import uvicorn

# Environment configuration
PORT = int(os.getenv("PORT", 8000))
ENV = os.getenv("ENVIRONMENT", "production")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TryOnDiffusion API", 
    version="1.0.0",
    description="Virtual Try-On API (Mock Mode for Deployment Testing)"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """Load and convert uploaded image to PIL Image"""
    try:
        # Read the file content
        contents = upload_file.file.read()
        
        # Reset file pointer for potential reuse
        upload_file.file.seek(0)
        
        # Create BytesIO object and open with PIL
        image_bytes = io.BytesIO(contents)
        image = Image.open(image_bytes)
        
        # Ensure image is loaded by calling load()
        image.load()
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Error loading image from upload: {e}")
        logger.error(f"Upload file type: {type(upload_file)}")
        logger.error(f"Content type: {getattr(upload_file, 'content_type', 'unknown')}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def create_enhanced_mock_tryon(person_img: Image.Image, garment_img: Image.Image) -> Image.Image:
    """Create an enhanced mock try-on result"""
    try:
        # Resize images to consistent size
        size = (512, 512)
        person_resized = person_img.resize(size, Image.Resampling.LANCZOS)
        garment_resized = garment_img.resize(size, Image.Resampling.LANCZOS)
        
        # Create base result with person image
        result = person_resized.copy()
        
        # Create a mask for the clothing area (center region)
        mask = Image.new('L', size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Draw clothing area (torso region)
        mask_draw.ellipse([
            size[0]//4, size[1]//3,  # top-left
            3*size[0]//4, 3*size[1]//4  # bottom-right
        ], fill=255)
        
        # Blur the mask for smooth blending
        mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
        
        # Resize garment to fit clothing area
        garment_fitted = garment_resized.resize((size[0]//2, size[1]//2), Image.Resampling.LANCZOS)
        
        # Create a new image for the garment overlay
        garment_overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        
        # Position garment in center
        paste_x = (size[0] - garment_fitted.width) // 2
        paste_y = size[1]//3
        garment_overlay.paste(garment_fitted, (paste_x, paste_y))
        
        # Convert to RGB for blending
        garment_overlay = garment_overlay.convert('RGB')
        
        # Blend using the mask
        result = Image.composite(garment_overlay, result, mask)
        
        # Add a subtle border to show it's a mock
        draw = ImageDraw.Draw(result)
        draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(0, 255, 0), width=3)
        
        # Add text watermark
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((20, 20), "MOCK TRYON", fill=(0, 255, 0), font=font)
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating mock result: {e}")
        # Fallback to simple blend
        size = (256, 256)
        person_simple = person_img.resize(size, Image.Resampling.LANCZOS)
        garment_simple = garment_img.resize(size, Image.Resampling.LANCZOS)
        return Image.blend(person_simple, garment_simple, 0.3)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TryOnDiffusion API is running!",
        "status": "healthy",
        "mode": "mock",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_status": "ready",
        "device": "cpu",
        "mode": "mock",
        "message": "TryOnDiffusion API is running! (Mock mode for deployment testing)",
        "port": PORT,
        "environment": ENV
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
    Virtual try-on endpoint (Enhanced Mock Implementation)
    """
    
    try:
        logger.info("Processing virtual try-on request (mock mode)...")
        
        # Validate file types
        if not person_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Person file must be an image")
        if not garment_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Garment file must be an image")
        
        # Load images
        person_img = load_image_from_upload(person_image)
        garment_img = load_image_from_upload(garment_image)
        
        logger.info(f"Loaded images - Person: {person_img.size}, Garment: {garment_img.size}")
        
        # Create enhanced mock result
        result_image = create_enhanced_mock_tryon(person_img, garment_img)
        
        # Convert to base64 for response
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG", quality=95)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("Virtual try-on completed successfully! (Mock mode)")
        
        return {
            "success": True,
            "result_image": f"data:image/png;base64,{image_base64}",
            "message": "Virtual try-on completed successfully! (Mock mode - enhanced blending)",
            "mock_mode": True,
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
            "processing_time": "~2-3 seconds (mock)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in virtual try-on: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")

@app.post("/tryon_simple")
async def virtual_tryon_simple(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    """Simplified virtual try-on endpoint with default parameters"""
    return await virtual_tryon(
        person_image=person_image,
        garment_image=garment_image,
        clothing_agnostic_image=None,
        guidance_scale=2.0,
        num_steps=20
    )

# Main entry point
if __name__ == "__main__":
    logger.info(f"Starting TryOnDiffusion API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

# For direct uvicorn import
# This allows: uvicorn main:app
# No need for complex module paths 