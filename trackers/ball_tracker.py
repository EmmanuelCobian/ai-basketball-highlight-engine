from ultralytics import YOLO
import pandas as pd
import math
import sys
sys.path.append('../')
import config
from utils import get_bbox_center, get_bbox_height, get_bbox_width
from .streaming_tracker import StreamingTracker

class BallTracker(StreamingTracker):
    """
    A streaming ball and hoop tracker that processes frames one at a time.
    """
    
    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.ball_tracks_history = []
        self.hoop_tracks_history = []
        self.frame_count = 0
        
    def process_frame(self, frame):
        """
        Process a single frame for ball and hoop tracking.
        
        Args:
            frame: Video frame as numpy array.
            
        Returns:
            dict: Ball and hoop tracking results for this frame.
        """
        self.frame_count += 1
        detection = self.model.predict([frame], verbose=False)[0]
        
        ball_frame_tracks = {}
        hoop_frame_tracks = {}
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        for box_id, box in enumerate(detection.boxes):
            ball_candidates = []
            if int(box.cls) == cls_names_inv[config.ball_label]:
                bbox = box.xyxy.cpu().numpy().flatten().tolist()
                conf = float(box.conf)
                ball_candidates.append({'conf': conf, 'bbox': bbox})
            if int(box.cls) == cls_names_inv[config.hoop_label]:
                bbox = box.xyxy.cpu().numpy().flatten().tolist()
                hoop_frame_tracks[box_id] = {
                    'bbox': bbox,
                    'bbox_center': get_bbox_center(bbox),
                    'bbox_width': get_bbox_width(bbox),
                    'bbox_height': get_bbox_height(bbox),
                    'frame': self.frame_count,
                }

        if ball_candidates:
            best_ball = max(ball_candidates, key=lambda x: x['conf'])
            bbox = best_ball['bbox']
            if not self.wrong_detection(bbox):
                ball_frame_tracks[1] = {
                    'bbox': bbox,
                    'bbox_center': get_bbox_center(bbox),
                    'bbox_width': get_bbox_width(bbox),
                    'bbox_height': get_bbox_height(bbox),
                    'frame': self.frame_count,
                }
                self.ball_tracks_history.append(ball_frame_tracks)
        
        if len(hoop_frame_tracks) != 0:
            self.hoop_tracks_history.append(hoop_frame_tracks)
        
        return ball_frame_tracks, hoop_frame_tracks
    
    def get_hoop_tracks_history(self):
        """Get all hoop tracking results so far."""
        return self.hoop_tracks_history
    
    def get_ball_tracks_history(self):
        """Get all ball tracking results so far."""
        return self.ball_tracks_history
    
    def wrong_detection(self, cur_bbox, max_frame_gap=5, lookback_frames=3):
        """
        Detect if a detection is valid based on consistency with recent ball positions.
        Args:
            cur_bbox (list): Current ball bbox.
            max_frame_gap (int): Maximum frame gap to consider for validation.
            lookback_frames (int): Number of recent frames to validate against.
        Returns:
            bool: True if detection is likely wrong, False otherwise.
        """
        history = self.get_ball_tracks_history()
        if len(history) == 0:
            return False
        
        x2, y2 = get_bbox_center(cur_bbox)
        recent_tracks = []
        for track_frame in reversed(history[-lookback_frames:]):
            track = track_frame[1]
            f_diff = self.frame_count - track['frame']
            if f_diff <= max_frame_gap:
                recent_tracks.append(track)
        
        if not recent_tracks:
            return False
        
        for track in recent_tracks:
            x1, y1 = track['bbox_center']
            w1, h1 = track['bbox_width'], track['bbox_height']
            
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            max_dist = 3 * math.sqrt(w1 ** 2 + h1 ** 2)
            
            if dist <= max_dist:
                return False
        return True
    
    def interpolate_ball_position(self):
        """
        Interpolate missing ball position. Uses the ball history tracks to interpolate the next ball bbox in the sequence
        """
        ball_tracks = self.get_ball_tracks_history()
        df_list = [track.get(1, {}).get("bbox", []) for track in ball_tracks]
        df_list.append([None, None, None, None])
        
        df = pd.DataFrame(df_list, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        
        return df.iloc[-1].tolist()
