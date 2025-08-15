"""
Basketball Analysis API - Application Entry Point

This file serves as the main entry point that imports the FastAPI app
from our modular api package.
"""
from api.main import app

# Re-export the app for uvicorn and other tools
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
