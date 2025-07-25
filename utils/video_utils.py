"""
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save
processed frames back to video files, with support for common video formats.
"""

import cv2
import os

def read_video(video_path):
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of video frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def stream_video_frames(video_path):
    """
    Generator that yields video frames one at a time for streaming processing.

    Args:
        video_path (str): Path to the input video file.

    Yields:
        tuple: (frame_number, frame) where frame is a numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_num, frame
        frame_num += 1
    
    cap.release()

def get_video_info(video_path):
    """
    Get video properties like frame count, fps, and dimensions.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        dict: Dictionary containing video properties.
    """
    cap = cv2.VideoCapture(video_path)
    
    properties = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    return properties

def save_video(ouput_video_frames,output_video_path):
    """
    Save a sequence of frames as a video file.

    Creates necessary directories if they don't exist and writes frames using XVID codec.

    Args:
        ouput_video_frames (list): List of frames to save.
        output_video_path (str): Path where the video should be saved.
    """
    # If folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore[attr-defined]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()

class StreamingVideoWriter:
    """
    A video writer class for streaming frame-by-frame video processing.
    """
    
    def __init__(self, output_path, width, height, fps=24.0):
        """
        Initialize the streaming video writer.
        
        Args:
            output_path (str): Path where the video should be saved.
            width (int): Video frame width.
            height (int): Video frame height.
            fps (float): Frames per second.
        """
        # Create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    def write_frame(self, frame):
        """
        Write a single frame to the video.
        
        Args:
            frame: Video frame as numpy array.
        """
        self.writer.write(frame)
        
    def release(self):
        """
        Release the video writer and finalize the video file.
        """
        self.writer.release()