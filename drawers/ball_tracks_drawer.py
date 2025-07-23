from .utils import draw_traingle
import cv2
from utils import get_bbox_center

class BallTracksDrawer:
    """
    A drawer class responsible for drawing ball tracks on video frames

    Attributes:
        ball_pointer_color (tuple): the color used to draw the ball pointers (RGB)
    """
    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """
        Draws ball pointers on each video frame based on tracking information

        Args:
            video_frames (list): a list of video frames
            tracks (list): a list of dictionaries where each dictionary contains ball information for the corresponding frame

        Returns:
            list: a list of processed video frames with drawn ball pointers
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            output_frame = frame.copy()
            ball_dict = tracks[frame_num]

            for _, ball in ball_dict.items():
                if ball['bbox'] is None:
                    continue
                output_frame = draw_traingle(frame, ball['bbox'], self.ball_pointer_color)
            
            output_video_frames.append(output_frame)
        return output_video_frames