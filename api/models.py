"""
Pydantic models and type definitions for the Basketball Analysis API.

This module contains all request/response models and type hints
used throughout the API.
"""
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class UploadURLResponse(BaseModel):
    """Response model for the upload URL endpoint."""
    session_id: str = Field(description="Unique session identifier")
    upload_url: str = Field(description="Pre-signed S3 URL for uploading video")
    s3_key: str = Field(description="S3 object key where video will be stored")
    metadata: Dict[str, str] = Field(description="Metadata required for S3 upload")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "upload_url": "https://bucket.s3.amazonaws.com/path/to/upload?signature=...",
                "s3_key": "temp-uploads/session-id/video.mp4",
                "metadata": {
                    "original-filename": "basketball_game.mp4",
                    "upload-timestamp": "1640995200",
                    "session-id": "123e4567-e89b-12d3-a456-426614174000"
                }
            }
        }


class StartProcessingRequest(BaseModel):
    """Request model for starting video processing."""
    s3_key: str = Field(description="S3 object key of the uploaded video")

    class Config:
        schema_extra = {
            "example": {
                "s3_key": "temp-uploads/session-id/video.mp4"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(default="ok", description="Service health status")
    timestamp: str = Field(description="Current server timestamp")


# =============================================================================
# WEBSOCKET MESSAGE MODELS
# =============================================================================

class WebSocketStatusUpdate(BaseModel):
    """Status update message sent over WebSocket."""
    type: str = Field(default="status_update")
    frame_num: int = Field(description="Current frame number being processed")
    frame_total: int = Field(description="Total number of frames in video")
    fps: float = Field(description="Processing frames per second")
    message: str = Field(description="Human-readable status message")


class WebSocketUserInput(BaseModel):
    """User input request sent over WebSocket."""
    type: str = Field(default="user_input_required")
    input_type: str = Field(description="Type of input required")
    frame_num: int = Field(description="Frame number where input is needed")
    data: Dict[str, Any] = Field(description="Input-specific data")


class WebSocketError(BaseModel):
    """Error message sent over WebSocket."""
    type: str = Field(default="error")
    message: str = Field(description="Error description")
    frame_num: int = Field(description="Frame number where error occurred")
    fps: float = Field(description="Processing FPS at time of error")


class WebSocketCompletion(BaseModel):
    """Completion message sent over WebSocket."""
    type: str = Field(default="completed")
    frame_num: int = Field(description="Final frame number")
    fps: float = Field(description="Final processing FPS")
    summary: Dict[str, Any] = Field(description="Processing summary and results")


# =============================================================================
# PROCESSING DATA TYPES
# =============================================================================

# Type aliases for better readability
PlayerTrack = Dict[int, Dict[str, Any]]
PlayerSuggestion = Tuple[int, float]
SessionData = Dict[str, Any]

# Player data structure for API responses
class PlayerInfo(BaseModel):
    """Information about a detected player."""
    id: int = Field(description="Player identifier")
    bbox: Optional[List[float]] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    center: Optional[List[float]] = Field(description="Center point coordinates [x, y]")
    confidence: float = Field(description="Detection confidence score")


# Session state enumeration
class SessionStatus:
    """Valid session states."""
    CREATED = "created"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


# User input types
class InputType:
    """Types of user input that can be requested."""
    PLAYER_SELECTION = "player_selection"
    CONFIRMATION = "confirmation"
    REASSIGNMENT_SELECTION = "reassignment_selection"


# WebSocket message types
class MessageType:
    """Types of WebSocket messages."""
    STATUS_UPDATE = "status_update"
    USER_INPUT_REQUIRED = "user_input_required"
    ERROR = "error"
    COMPLETED = "completed"
    HEARTBEAT = "heartbeat"
