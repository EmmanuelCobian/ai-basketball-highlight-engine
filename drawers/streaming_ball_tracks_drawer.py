from .utils import draw_box
import cv2

class StreamingBallTracksDrawer:
    """
    A streaming drawer class for drawing ball tracks on individual frames.
    """
    
    def __init__(self):
        pass
    
    def draw_frame(self, frame, ball_track):
        """
        Draw ball tracks on a single frame.
        
        Args:
            frame: Video frame as numpy array.
            ball_track: Ball tracking results for this frame.
            
        Returns:
            Processed frame with drawn ball tracks.
        """
        frame = frame.copy()
        
        for track_id, ball in ball_track.items():
            frame = draw_box(frame, ball['bbox'], (0, 255, 0))
        
        return frame
