from ultralytics import YOLO
import sys
sys.path.append('../')
import config
from utils import get_bbox_height, get_bbox_center, get_bbox_width
from .streaming_tracker import StreamingTracker

class StreamingHoopTracker(StreamingTracker):
    """
    A streaming hoop tracker that processes frames one at a time.
    """
    
    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.tracks_history = []
        
    def process_frame(self, frame):
        """
        Process a single frame for hoop tracking.
        
        Args:
            frame: Video frame as numpy array.
            
        Returns:
            dict: Hoop tracking results for this frame.
        """
        detection = self.model.predict([frame])[0]
        
        frame_tracks = {}
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        for box_id, box in enumerate(detection.boxes):
            if int(box.cls) == cls_names_inv[config.hoop_label]:
                bbox = box.xyxy.cpu().numpy().flatten().tolist()
                frame_tracks[box_id] = {
                    'bbox': bbox,
                    'bbox_center': get_bbox_center(bbox),
                    'bbox_width': get_bbox_width(bbox),
                    'bbox_height': get_bbox_height(bbox),
                }
        
        self.tracks_history.append(frame_tracks)
        
        return frame_tracks
    
    def get_tracks_history(self):
        """Get all tracking results so far."""
        return self.tracks_history
