from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import base64
import traceback
from typing import Optional
import logging
import os

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

# Import TryOnDiffusion
from tryondiffusion import TryOnImagen, get_unet_by_name

app = FastAPI(title="TryOnDiffusion API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the model
imagen_model = None
device = None

def initialize_model():
    """Initialize the TryOnDiffusion model"""
    global imagen_model, device
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Create UNets
        logger.info("Creating UNets...")
        unet1 = get_unet_by_name("base")
        unet2 = get_unet_by_name("sr")
        
        # Create Imagen model
        imagen_model = TryOnImagen(
            unets=(unet1, unet2),
            image_sizes=((128, 128), (256, 256)),
            timesteps=(64, 32),  # Reduced timesteps for faster inference
        )
        imagen_model = imagen_model.to(device)
        
        logger.info("TryOnDiffusion model initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """Load and convert uploaded image to PIL Image"""
    contents = upload_file.file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def image_to_tensor(image: Image.Image, size: tuple = (256, 256)) -> torch.Tensor:
    """Convert PIL Image to tensor"""
    # Resize image
    image = image.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor and rearrange dimensions (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    
    # Clamp values and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    image_array = tensor.permute(1, 2, 0).numpy()
    
    # Convert to PIL Image
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    
    return image

def generate_dummy_pose(batch_size: int = 1) -> torch.Tensor:
    """Generate dummy pose data (18 keypoints with x,y coordinates)"""
    # For simplicity, generate random pose data
    # In a real implementation, you'd extract pose from the person image
    pose = torch.randn(batch_size, 18, 2)
    return pose

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("Initializing TryOnDiffusion model...")
    success = initialize_model()
    if not success:
        logger.error("Failed to initialize model on startup")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "TryOnDiffusion API is running!"}

@app.get("/health")
async def health():
    """Health check with model status"""
    model_status = "ready" if imagen_model is not None else "not_initialized"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": device,
        "message": "TryOnDiffusion API is running!"
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
    Virtual try-on endpoint
    
    - **person_image**: Image of the person/model
    - **garment_image**: Image of the clothing item to try on
    - **clothing_agnostic_image**: Optional clothing-agnostic person image
    - **guidance_scale**: Controls how closely the model follows the input (higher = more faithful)
    - **num_steps**: Number of denoising steps (higher = better quality but slower)
    """
    
    if imagen_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        logger.info("Processing virtual try-on request...")
        
        # Load images
        person_img = load_image_from_upload(person_image)
        garment_img = load_image_from_upload(garment_image)
        
        # Use clothing-agnostic image if provided, otherwise use person image
        if clothing_agnostic_image:
            ca_img = load_image_from_upload(clothing_agnostic_image)
        else:
            ca_img = person_img  # Fallback to person image
        
        logger.info(f"Loaded images - Person: {person_img.size}, Garment: {garment_img.size}")
        
        # Convert images to tensors
        person_tensor = image_to_tensor(person_img, (256, 256)).to(device)
        garment_tensor = image_to_tensor(garment_img, (256, 256)).to(device)
        ca_tensor = image_to_tensor(ca_img, (256, 256)).to(device)
        
        # Generate dummy poses (in a real implementation, extract from images)
        person_pose = generate_dummy_pose(1).to(device)
        garment_pose = generate_dummy_pose(1).to(device)
        
        logger.info("Running inference...")
        
        # Run inference
        with torch.no_grad():
            images = imagen_model.sample(
                ca_images=ca_tensor,
                garment_images=garment_tensor,
                person_poses=person_pose,
                garment_poses=garment_pose,
                batch_size=1,
                cond_scale=guidance_scale,
                start_at_unet_number=1,
                return_all_unet_outputs=False,
                return_pil_images=False,  # We'll convert manually
                use_tqdm=False,
                use_one_unet_in_gpu=True,
            )
        
        # Convert result to PIL Image
        result_image = tensor_to_image(images)
        
        # Convert to base64 for response
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("Virtual try-on completed successfully!")
        
        return {
            "success": True,
            "result_image": f"data:image/png;base64,{image_base64}",
            "message": "Virtual try-on completed successfully!"
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