from .utils import draw_box

class HoopTracksDrawer:
    """
    A drawer class responsible for drawing hoop tracks on video frames

    Attributes:
        hoop_border_color (tuple): the color used to draw the ball pointers (RGB)
    """
    def __init__(self):
        self.hoop_border_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """
        Draws hoop bounding boxes on each video frame based on tracking information

        Args:
            video_frames (list): a list of video frames
            tracks (list): a list of dictionaries where each dictionary contains ball information for the corresponding frame

        Returns:
            list: a list of processed video frames with drawn hoop pointers
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            output_frame = frame.copy()
            hoop_dict = tracks[frame_num]

            for _, hoop in hoop_dict.items():
                if hoop['bbox'] is None:
                    continue
                output_frame = draw_box(frame, hoop['bbox'], self.hoop_border_color)
            
            output_video_frames.append(output_frame)
        return output_video_frames