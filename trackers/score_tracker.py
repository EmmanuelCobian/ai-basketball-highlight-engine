from utils import detect_down, detect_up, score

class ScoreTracker:
    """
    Tracks basketball shot attempts and makes based on ball and hoop tracking data.

    Attributes:
        makes (int): Number of successful shots detected.
        attempts (int): Number of shot attempts detected.
        up (bool): Flag indicating if the ball is moving upwards (potential shot start).
        down (bool): Flag indicating if the ball is moving downwards (potential shot end).
        up_frame (int): Frame index when upward movement is detected.
        down_frame (int): Frame index when downward movement is detected.

    Methods:
        get_scores(ball_tracks, hoop_tracks):
            Processes sequences of ball and hoop tracking data to detect shot attempts and makes.
            Returns a list of tuples containing cumulative makes and attempts for each frame.
    """
    def __init__(self):
        self.makes = self.attempts = 0
        self.up = self.down = False
        self.up_frame = self.down_frame = 0

    def get_scores(self, ball_tracks, hoop_tracks):
        """
        Processes ball and hoop tracking data to detect shot attempts and makes.

        Args:
            ball_tracks (list): Sequence of ball tracking data per frame.
            hoop_tracks (list): Sequence of hoop tracking data per frame.

        Returns:
            list: Tuples of (makes, attempts) for each frame.
        """
        tracks = []
        for i in range(len(ball_tracks)):
            if not self.up:
                self.up = detect_up(ball_tracks[i], hoop_tracks[i])
                if self.up:
                    self.up_frame = i

            if self.up and not self.down:
                self.down = detect_down(ball_tracks[i], hoop_tracks[i])
                if self.down:
                    self.down_frame = i

            if i % 5 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = self.down = False

                    if score(ball_tracks, hoop_tracks, i):
                        self.makes += 1

            tracks.append((self.makes, self.attempts))

        return tracks
