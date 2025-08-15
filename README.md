# Basketball Analysis Project

This project analyzes basketball videos to track players, detect ball possession, and identify highlights using YOLO models and computer vision techniques.

## Features

- **Real-time player tracking** with YOLO models
- **Ball possession detection** and attribution
- **Highlight detection** from predefined time intervals
- **Player substitution handling** when tracking is lost
- **Live video visualization** with tracking overlays
- **Frame-by-frame analysis** with pause/resume controls

## Project Structure

- `main.py` - Main application entry point
- `config.py` - Configuration settings
- `trackers/` - Player, ball, and hoop tracking modules
- `drawers/` - Visualization modules for drawing tracks
- `utils/` - Utility functions for video processing
- `ball_aquisition/` - Ball possession detection logic
- `training_notebooks/` - Jupyter notebooks for model training

## Prerequisites

- Python 3.8+
- A webcam or video files for analysis
- YOLO model files (see setup instructions below)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/EmmanuelCobian/ai-basketball-highlight-engine.git
cd ai-basketball-highlight-engine
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download/Prepare Model Files

You'll need YOLO model files for the application to work:

- **Player detection**: `yolo11s.pt` (place in root directory)
- **Ball/hoop detection**: `best_im.pt` (place in `models/` directory)

You can:
- Train your own models using the notebooks in `training_notebooks/`
- Download pre-trained YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Use the default YOLO models and retrain for basketball-specific detection

### 5. Prepare Video Files

Add your basketball video files to the `input_videos/` directory:
```bash
mkdir -p input_videos
# Copy your .mp4 or .mov files here
```

### 6. Configure Highlights (Optional)

Edit `highlights.txt` to define time intervals for highlight detection:
```
# Format: HH:MM:SS,HH:MM:SS
00:01:00,00:01::05
00:08:32,00:08::37
```

## Usage

### Basic Usage
```bash
python main.py
```

### Interactive Controls
- **'q'** - Quit the application
- **'p'** - Pause/resume playback
- **'s'** - Save screenshot of current frame

### Initial Setup
1. Run the application
2. When prompted, enter the player ID you want to track for highlights
3. The system will track that player throughout the video

## Dependencies

This project uses the following key libraries:

- **OpenCV** - Video processing and computer vision
- **NumPy** - Numerical computations
- **Ultralytics** - YOLO model implementation
- **Supervision** - Computer vision utilities
- **PyTorch** - Deep learning framework
- **Pandas** - Data manipulation

### Training Custom Models

Use the Jupyter notebooks in `training_notebooks/` to:
- `basketball_player_detection.ipynb` - Train player detection models
- `basketball_ball_detection.ipynb` - Train ball detection models

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Supervision](https://github.com/roboflow/supervision) for tracking utilities

## API (FastAPI + WebSocket)

This project exposes a streaming API for uploading a video, processing it frame-by-frame, pausing for user input via WebSocket messages, and returning a final JSON highlight summary (no video frames are streamed).

### Run the API server

- Install dependencies:
  - `pip install -r requirements.txt`
- Start the server:
  - `python -m uvicorn api_main:app --host 0.0.0.0 --port 8000`

Health check:
- GET `/health` → `{ "status": "ok" }`

### Processing flow overview

1) Upload a video (multipart/form-data):
- POST `/sessions`
  - Form field name: `file`
  - Response:
    ```json
    { "session_id": "<uuid>" }
    ```

2) Open a WebSocket for streaming status + interactive inputs:
- WS `/ws/{session_id}`
  - The server will:
    - Stream periodic `status_update` messages
    - Send `user_input_required` messages when selection/confirmation is needed
    - Finally send `completed` with a `summary` JSON and close the socket
    - On failures, send `error` then close the socket

Sessions are independent (MVP). Uploaded files are stored in a temporary session directory and cleaned up at the end of processing.

### Message contracts

All messages are JSON objects. Every server→client message now includes the current `frame_num` and the video `fps` (fps is constant across the session).

#### Server → Client

- Status update
  ```json
  {
    "type": "status_update",
    "frame_num": 511,
    "frame_total": 1000,
    "fps": 29.97,
    "message": "Processing frames..."
  }
  ```

- User input required: initial player selection
  ```json
  {
    "type": "user_input_required",
    "input_type": "player_selection",
    "frame_num": 42,
    "fps": 29.97,
    "data": {
      "available_players": [
        { "id": 1, "bbox": [x1,y1,x2,y2], "center": [x,y], "confidence": 0.88 },
        { "id": 5, "bbox": [x1,y1,x2,y2], "center": [x,y], "confidence": 0.74 }
      ],
      "message": "Select initial player to track"
    }
  }
  ```

- User input required: temporary assignment confirmation
  ```json
  {
    "type": "user_input_required",
    "input_type": "confirmation",
    "frame_num": 360,
    "fps": 29.97,
    "data": {
      "original_id": 7,
      "current_id": 13,
      "original_bbox": [x1,y1,x2,y2],
      "current_bbox": [x1,y1,x2,y2],
      "message": "Confirm temporary assignment: keep tracking 13 as permanent replacement for 7?"
    }
  }
  ```

- User input required: reassignment selection
  ```json
  {
    "type": "user_input_required",
    "input_type": "reassignment_selection",
    "frame_num": 512,
    "fps": 29.97,
    "data": {
      "available_players": [
        { "id": 1, "bbox": [x1,y1,x2,y2], "center": [x,y], "confidence": 0.88 },
        { "id": 5, "bbox": [x1,y1,x2,y2], "center": [x,y], "confidence": 0.74 }
      ],
      "current_tracked": { "id": 13, "bbox": [x1,y1,x2,y2] },
      "suggestions": [
        { "id": 9, "confidence": 0.72, "bbox": [x1,y1,x2,y2] },
        { "id": 2, "confidence": 0.65, "bbox": [x1,y1,x2,y2] }
      ],
      "message": "Choose a player to reassign or continue without tracking (choice 0)"
    }
  }
  ```

- Completed summary
  ```json
  {
    "type": "completed",
    "frame_num": 1234,
    "fps": 29.97,
    "summary": {
      "processed_frames": 1234,
      "tracked_player_ids": [7, 13],
      "tracked_player_highlights": 3,
      "total_highlights": 5,
      "highlights": [
        {
          "interval": [100, 240],
          "possessions": { "7": 45, "12": 18 },
          "winner": { "player_id": 7, "frames": 45 },
          "tracked_player_won": true
        }
      ]
    }
  }
  ```

- Error
  ```json
  { "type": "error", "message": "<details>", "frame_num": 42, "fps": 29.97 }
  ```

#### Client → Server (WebSocket replies)

- Player selection
  ```json
  { "response_type": "player_selection", "player_id": 7 }
  ```

- Confirmation (yes/no)
  ```json
  { "response_type": "confirmation", "confirmed": true }
  ```

- Reassignment selection
  - Choose by player id:
    ```json
    { "response_type": "reassignment_selection", "player_id": 9 }
    ```
  - Choose by suggestion index (1-based):
    ```json
    { "response_type": "reassignment_selection", "suggestion_index": 1 }
    ```
  - Continue without tracking:
    ```json
    { "response_type": "reassignment_selection", "choice": 0 }
    ```

### Client example

A minimal Python client is provided in `client_example.py` which:
- Uploads a video via `POST /sessions`
- Connects to `WS /ws/{session_id}`
- Auto-responds to input prompts (for demo)
- Prints the final summary JSON

Usage:
```bash
python client_example.py input_videos/video_1.mp4
```

### Notes
- Only status updates are streamed; no video frames are sent.
- Highlight windows are generated per session in-memory by the highlight engine; no shared highlight file is used at runtime.
- CORS is configured to allow all origins by default for development. Restrict `allow_origins` for production.
- This MVP keeps session state in memory and cleans up temp files at the end of each session.
