#!/usr/bin/env python3
"""
Startup script for TryOnDiffusion API
"""

import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the TryOnDiffusion API"""
    
    # Print startup banner
    print("=" * 60)
    print("🎨 MODILAI - TryOnDiffusion API Server")
    print("=" * 60)
    print("Starting virtual try-on service...")
    print("API will be available at: http://localhost:8001")
    print("Documentation: http://localhost:8001/docs")
    print("=" * 60)
    
    try:
        # Import and run the API
        from tryon_api import app
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            reload=False  # Set to True for development
        )
        
    except ImportError as e:
        logger.error(f"Failed to import TryOnDiffusion modules: {e}")
        logger.error("Make sure all dependencies are installed:")
        logger.error("pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 