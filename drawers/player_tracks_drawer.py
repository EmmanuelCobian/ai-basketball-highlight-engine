from .utils import draw_ellipse, draw_traingle
import cv2

class PlayerTracksDrawer:
    """
    A drawer class responsible for drawing player tracks on video frames
    """
    def __init__(self):
        return
    
    def draw(self, video_frames, tracks, ball_aquisition):
        """
        Draws nba2k like circles around players based on tracker infomation

        Args:
            video_frames (list): list of video frames
            tracks (list): a list of dictionaries where each dictionary contains player information for each frame

        Returns:
            list: a list of processed video frames with drawn player circles
        
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks[frame_num]
            player_id_with_ball = ball_aquisition[frame_num]

            for global_id, player in player_dict.items():
                local_id = player.get("local_id")

                if global_id == player_id_with_ball:
                    frame = draw_traingle(frame, player['bbox'], (0, 0, 255))

                label = f"G:{global_id} / L:{local_id}" if local_id is not None else f"G:{global_id}"
                frame = draw_ellipse(frame, player['bbox'], (255, 255, 255), str(local_id))

            output_video_frames.append(frame)
        
        return output_video_frames