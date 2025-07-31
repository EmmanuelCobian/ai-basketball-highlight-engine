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
# .venv\Scripts\activate
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
# Format: start_time_seconds,end_time_seconds
10.5,25.0
45.2,60.8
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
