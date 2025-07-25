import sys
sys.path.append('../')
from utils import get_bbox_center, measure_distance
from .streaming_tracker import StreamingTracker

class StreamingScoreTracker(StreamingTracker):
    """
    A streaming score tracker that processes frames one at a time.
    """
    
    def __init__(self):
        super().__init__()
        self.scores_history = []
        self.makes = self.attempts = 0
        self.up = self.down = False
        self.up_frame = self.down_frame = 0
        
    def process_frame(self, ball_track, hoop_track):
        """
        Process a single frame for score tracking.
        
        Args:
            ball_track: Ball tracking results for this frame.
            hoop_track: Hoop tracking results for this frame.
            
        Returns:
            int: Score change for this frame (0 or 1).
        """
        score = 0
        
        if 1 in ball_track and len(hoop_track) > 0:
            ball_bbox = ball_track[1]['bbox']
            
            for hoop_id, hoop in hoop_track.items():
                hoop_bbox = hoop['bbox']
                
                ball_center = get_bbox_center(ball_bbox)
                distance = measure_distance(ball_center, get_bbox_center(hoop_bbox))
                
                # Simple scoring logic - if ball is very close to hoop center
                if distance < 50:  # Threshold in pixels
                    score = 1
                    break
        
        self.scores_history.append(score)
        self.frame_count += 1
        
        return score
    
    def get_scores_history(self):
        """Get all score results so far."""
        return self.scores_history
