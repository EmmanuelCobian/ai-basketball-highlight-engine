"""
Configuration settings for the Basketball Analysis API.

This module contains all configuration constants, environment variables,
and settings used throughout the application.
"""
import os
from typing import Final

# =============================================================================
# MODEL PATHS
# =============================================================================
PLAYER_MODEL_PATH: Final[str] = "yolo11s.pt"
BALL_MODEL_PATH: Final[str] = "models/best_im.pt"

# =============================================================================
# AWS CONFIGURATION
# =============================================================================
S3_BUCKET: Final[str] = os.getenv("S3_BUCKET", "goplai-s3")
AWS_REGION: Final[str] = os.getenv("AWS_REGION", "us-east-2")

# =============================================================================
# WEBSOCKET & TIMEOUT CONFIGURATION
# =============================================================================
# How long the entire WebSocket connection can stay open
WEBSOCKET_TIMEOUT: Final[int] = 3600  # 1 hour

# How often to send heartbeat messages to keep connection alive
HEARTBEAT_INTERVAL: Final[int] = 30   # 30 seconds

# Maximum gap allowed between processing updates before considering it stalled
MAX_PROCESSING_GAP: Final[int] = 60   # 1 minute

# How long to wait for user input before timing out
USER_INPUT_TIMEOUT: Final[int] = 300  # 5 minutes

# How long to wait for initial connection establishment
CONNECTION_TIMEOUT: Final[int] = 30   # 30 seconds

# =============================================================================
# API CONFIGURATION
# =============================================================================
# CORS settings
ALLOWED_ORIGINS = ["*"]  # In production, specify actual origins

# API metadata
API_TITLE = "Basketball Highlight Engine API"
API_DESCRIPTION = """
Basketball Analysis API for processing videos and generating highlights.

This API provides endpoints to:
- Upload basketball videos to S3
- Process videos with AI-powered player and ball tracking
- Generate highlight reels
- Real-time WebSocket communication during processing
"""
API_VERSION = "1.0.0"
