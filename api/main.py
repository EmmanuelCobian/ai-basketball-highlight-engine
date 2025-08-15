"""
Basketball Highlight Engine API - Main Application

This is the main FastAPI application that provides endpoints for:
1. Video upload URL generation (S3 pre-signed URLs)
2. Video processing initiation
3. Real-time WebSocket communication during processing
4. Health monitoring

The application follows a modular architecture with clear separation of concerns:
- config.py: Configuration and constants
- models.py: Pydantic models and type definitions  
- services/: Business logic services (S3, WebSocket, Video Processing)
"""
import datetime
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.config import (
    API_TITLE, API_DESCRIPTION, API_VERSION, ALLOWED_ORIGINS,
    S3_BUCKET
)
from api.models import (
    UploadURLResponse, StartProcessingRequest, HealthResponse,
    SessionStatus
)
from api.services import s3_service, websocket_service, video_service


# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (in production, use Redis or database)
active_sessions: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/upload-url", response_model=UploadURLResponse)
async def get_upload_url(filename: str = Query(..., description="Name of the video file to upload")):
    """
    Generate a pre-signed S3 URL for video upload.
    
    This endpoint creates a new session and returns:
    - Pre-signed S3 URL for direct upload from client
    - Session ID for tracking the processing request
    - S3 key where the video will be stored
    - Required metadata for the upload
    
    Args:
        filename: Original name of the video file
        
    Returns:
        Upload URL response with session details
        
    Example:
        POST /upload-url?filename=basketball_game.mp4
        
        Response:
        {
            "session_id": "123e4567-e89b-12d3-a456-426614174000",
            "upload_url": "https://bucket.s3.amazonaws.com/...",
            "s3_key": "temp-uploads/session-id/video.mp4",
            "metadata": {...}
        }
    """
    try:
        session_id, upload_url, s3_key, metadata = s3_service.generate_upload_url(filename)
        active_sessions[session_id] = {
            "session_id": session_id,
            "s3_key": s3_key,
            "original_filename": filename,
            "status": SessionStatus.CREATED,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        return UploadURLResponse(
            session_id=session_id,
            upload_url=upload_url,
            s3_key=s3_key,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/sessions/{session_id}/start")
async def start_processing(session_id: str, request: StartProcessingRequest):
    """
    Start processing a video that has been uploaded to S3.
    
    This endpoint initiates the basketball analysis pipeline for a video
    that has been uploaded via the pre-signed URL from /upload-url.
    
    Args:
        session_id: Session identifier from upload URL response
        request: Processing request containing S3 key
        
    Returns:
        Success confirmation
        
    Raises:
        HTTPException: If session not found or processing fails to start
        
    Example:
        POST /sessions/{session_id}/start
        {
            "s3_key": "temp-uploads/session-id/video.mp4"
        }
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        if session["s3_key"] != request.s3_key:
            raise HTTPException(status_code=400, detail="S3 key mismatch")
        
        try:
            s3_service.s3_client.head_object(
                Bucket=S3_BUCKET, 
                Key=request.s3_key
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")
        
        session["status"] = SessionStatus.PROCESSING
        session["started_at"] = datetime.datetime.utcnow().isoformat()
        
        return {"message": "Processing started", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time video processing communication.
    
    This endpoint provides real-time updates during video processing including:
    - Processing status updates (frame progress, FPS)
    - User input requests (player selection, confirmations)
    - Error notifications
    - Processing completion with results
    - Heartbeat messages to keep connection alive
    
    Args:
        websocket: WebSocket connection
        session_id: Session identifier
        
    Message Types Sent:
        - status_update: Processing progress information
        - user_input_required: Request for user interaction
        - error: Error notifications
        - completed: Processing finished with results
        - heartbeat: Keep-alive messages
        
    Message Types Received:
        - User responses to input requests (JSON format)
        
    Example Connection:
        ws://localhost:8000/ws/{session_id}
    """
    try:
        if session_id not in active_sessions:
            await websocket.close(code=1008, reason="Invalid session_id")
            return
        
        session = active_sessions[session_id]
        if session["status"] != SessionStatus.PROCESSING:
            await websocket.close(code=1008, reason="Session not in processing state")
            return
        
        await websocket_service.connect(websocket, session_id)
        try:
            await video_service.process_video_session(session, websocket)
            session["status"] = SessionStatus.COMPLETED
            session["completed_at"] = datetime.datetime.utcnow().isoformat()
            
        except Exception as e:
            session["status"] = SessionStatus.ERROR
            session["error"] = str(e)
            session["error_at"] = datetime.datetime.utcnow().isoformat()
            
            await websocket_service.send_error(websocket, f"Processing failed: {str(e)}")
            raise
        
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error for session {session_id}: {e}")
    finally:
        websocket_service.disconnect(session_id)


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns:
        Service health status and timestamp
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.datetime.utcnow().isoformat()
    )


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Basic API information and available endpoints
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": "Basketball analysis API for processing videos and generating highlights",
        "endpoints": {
            "upload_url": "POST /upload-url?filename={filename}",
            "start_processing": "POST /sessions/{session_id}/start",
            "websocket": "WS /ws/{session_id}",
            "health": "GET /health"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get current status of a processing session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session status and metadata
        
    Raises:
        HTTPException: If session not found
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "created_at": session["created_at"],
        "original_filename": session["original_filename"],
        "s3_key": session["s3_key"]
    }


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
