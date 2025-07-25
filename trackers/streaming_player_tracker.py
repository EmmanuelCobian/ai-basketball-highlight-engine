from utils import GlobalIDManager
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
import sys
sys.path.append('../')
import config
from utils import get_bbox_center, get_bbox_height, get_bbox_width
from .streaming_tracker import StreamingTracker

class StreamingPlayerTracker(StreamingTracker):
    """
    A streaming player tracker that processes frames one at a time.
    """
    
    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.track_embeddings = defaultdict(list)
        self.track_bboxes = defaultdict(dict)
        self.tracks_history = []
        
    def process_frame(self, frame):
        """processes a single frame for player tracking

        Args:
            frame (np.ndarray): video frame to process

        Returns:
            dict: player tracks for the frame
        """
        detection = self.model.predict([frame], classes=[0])[0]
        
        frame_tracks = {}
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        detection_supervision = sv.Detections.from_ultralytics(detection)
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
        
        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            local_id = frame_detection[4]
            
            if cls_id == cls_names_inv[config.player_label]:
                frame_tracks[local_id] = {
                    "bbox": bbox, 
                    "local_id": local_id, 
                    "bbox_center": get_bbox_center(bbox), 
                    "bbox_width": get_bbox_width(bbox), 
                    "bbox_height": get_bbox_height(bbox),
                }
        
        self.tracks_history.append(frame_tracks)
        
        return frame_tracks
    
    def get_tracks_history(self):
        """Get all tracking results so far."""
        return self.tracks_history
