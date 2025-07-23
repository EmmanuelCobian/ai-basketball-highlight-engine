from .utils import draw_box
import cv2
import numpy as np

class ScoreTracksDrawer:
    """
    A drawer class responsible for drawing basketball shot attempts and scores on video frames

    Attributes:
        hoop_border_color (tuple): the color used to draw the ball pointers (RGB)
    """
    def __init__(self):
        self.hoop_border_color = (0, 255, 0)
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

    def draw(self, video_frames, tracks):
        """
        Draws hoop bounding boxes on each video frame based on tracking information

        Args:
            video_frames (list): a list of video frames
            tracks (list): tracks[i] = (total baskets made, total shot attempts made)

        Returns:
            list: a list of processed video frames with drawn scores
        """
        output_video_frames = []
        prev = (0, 0)
        for frame_num, frame in enumerate(video_frames):
            output_frame = frame.copy()
            makes, attempts = tracks[frame_num]
            prev_makes, prev_attempts = prev
            if attempts > prev_attempts:
                if makes == prev_makes:
                    self.overlay_color = (255, 0, 0)
                else:
                    self.overlay_color = (0, 255, 0)
                self.fade_counter = self.fade_frames
            prev = (makes, attempts)


            text = str(makes) + " / " + str(attempts)
            cv2.putText(output_frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (225, 225, 225), 6)
            cv2.putText(output_frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

            if self.fade_counter > 0:
                alpha = 0.2 * (self.fade_counter / self.fade_frames)
                output_frame = cv2.addWeighted(output_frame, 1 - alpha, np.full_like(output_frame, self.overlay_color), alpha, 0)
                self.fade_counter -= 1

            output_video_frames.append(output_frame)
        return output_video_frames