from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import math
import sys
sys.path.append('../')
from utils import save_stub, read_stub, get_bbox_width, get_bbox_height, get_bbox_center
import config

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = config.batch_size
        detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            detections_batch = self.model.predict(batch_frames)
            detections += detections_batch

        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track the ball accross frames using supervision

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): whether to read in cached results
            stub_path (str): path to the cached file
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks 
            
        detections = self.detect_frames(frames)
        tracks = []

        for i, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v : k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv[config.ball_label]:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[i][1] = {"bbox" : chosen_bbox}
        
        save_stub(stub_path, tracks)

        return tracks
    
    def remove_wrong_detections(self, ball_positions, movement_multiplier=1.0):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        Args:
            ball_positions (list): List of detected ball positions across frames.
            movement_multiplier (float): Multiplier for bounding box diagonal to determine max movement.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_data = ball_positions[i].get(1)
            if not current_data:
                continue
            
            current_box = current_data.get('bbox', [])
            if len(current_box) < 4:
                continue  # Bad or incomplete detection

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_data = ball_positions[last_good_frame_index].get(1)
            if not last_data:
                continue

            last_box = last_data.get('bbox', [])
            if len(last_box) < 4:
                continue

            # Compute center points
            last_center = np.array(get_bbox_center(last_box))
            current_center = np.array(get_bbox_center(current_box))

            width = get_bbox_width(last_box)
            height = get_bbox_height(last_box)
            frame_gap = i - last_good_frame_index

            max_dist = movement_multiplier * np.sqrt(width**2 + height**2) * frame_gap

            if np.linalg.norm(current_center - last_center) > max_dist :
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1 : {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    