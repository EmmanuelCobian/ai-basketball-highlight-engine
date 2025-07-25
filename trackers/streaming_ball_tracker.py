from ultralytics import YOLO
import pandas as pd
import sys
sys.path.append('../')
import config
from utils import get_bbox_center, get_bbox_height, get_bbox_width
from .streaming_tracker import StreamingTracker

class StreamingBallTracker(StreamingTracker):
    """
    A streaming ball tracker that processes frames one at a time.
    """
    
    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.tracks_history = []
        
    def process_frame(self, frame):
        """
        Process a single frame for ball tracking.
        
        Args:
            frame: Video frame as numpy array.
            
        Returns:
            dict: Ball tracking results for this frame.
        """
        detection = self.model.predict([frame])[0]
        
        frame_tracks = {}
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        for box_id, box in enumerate(detection.boxes):
            if int(box.cls) == cls_names_inv[config.ball_label]:
                bbox = box.xyxy.cpu().numpy().flatten().tolist()
                frame_tracks[1] = {
                    'bbox': bbox,
                    'bbox_center': get_bbox_center(bbox),
                    'bbox_width': get_bbox_width(bbox),
                    'bbox_height': get_bbox_height(bbox),
                }
                break
        
        self.tracks_history.append(frame_tracks)
        
        return frame_tracks
    
    def get_tracks_history(self):
        """Get all tracking results so far."""
        return self.tracks_history
    
    def remove_wrong_detections(self, ball_tracks=None):
        """
        Remove wrong ball detections based on position consistency.
        Works on the stored tracks history if no tracks provided.
        """
        if ball_tracks is None:
            ball_tracks = self.tracks_history
            
        ball_tracks_df = pd.DataFrame(ball_tracks)
        ball_tracks_df = ball_tracks_df.interpolate()
        ball_tracks_df = ball_tracks_df.bfill()
        
        cleaned_tracks = []
        for index, row in ball_tracks_df.iterrows():
            frame_tracks = {}
            if not pd.isna(row[1]):
                frame_tracks[1] = row[1]
            cleaned_tracks.append(frame_tracks)
        
        if ball_tracks is None:
            self.tracks_history = cleaned_tracks
            
        return cleaned_tracks
    
    def interpolate_ball_positions(self, ball_tracks=None):
        """
        Interpolate missing ball positions.
        Works on the stored tracks history if no tracks provided.
        """
        if ball_tracks is None:
            ball_tracks = self.tracks_history
            
        ball_positions = [track.get(1, {}).get("bbox", []) for track in ball_tracks]
        
        df_list = []
        for position in ball_positions:
            if len(position) == 4:
                df_list.append(position)
            else:
                df_list.append([None, None, None, None])
        
        df = pd.DataFrame(df_list, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        
        interpolated_tracks = []
        for index, row in df.iterrows():
            frame_tracks = {}
            if not pd.isna(row['x1']):
                bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
                frame_tracks[1] = {'bbox': bbox}
            interpolated_tracks.append(frame_tracks)
        
        if ball_tracks is None:
            self.tracks_history = interpolated_tracks
            
        return interpolated_tracks
