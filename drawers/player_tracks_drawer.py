from .utils import draw_ellipse, draw_traingle
import cv2

class PlayerTracksDrawer:
    """
    A streaming drawer class for drawing player tracks on individual frames.
    """
    
    def __init__(self):
        pass
    
    def draw_frame(self, frame, player_track, ball_acquisition):
        """
        Draw player tracks on a single frame.
        
        Args:
            frame: Video frame as numpy array.
            player_track: Player tracking results for this frame.
            ball_acquisition: Ball acquisition result for this frame.
            
        Returns:
            Processed frame with drawn player tracks.
        """
        frame = frame.copy()
        
        for global_id, player in player_track.items():
            local_id = player.get("local_id")
            
            if global_id == ball_acquisition:
                frame = draw_traingle(frame, player['bbox'], (0, 0, 255))
            
            label = f"G:{global_id} / L:{local_id}" if local_id is not None else f"G:{global_id}"
            frame = draw_ellipse(frame, player['bbox'], (255, 255, 255), str(local_id))
        
        return frame
