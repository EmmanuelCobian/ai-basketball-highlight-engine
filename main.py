from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, HoopTracker, ScoreTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, HoopTracksDrawer, ScoreTracksDrawer
from ball_aquisition import BallAquisitionDetector

def main():
    print("=====READING VIDEO FRAMES=====")
    video_frames = read_video("input_videos/im_1.mov")

    print("=====TRACKING PLAYERS=====")
    player_tracker = PlayerTracker("yolo11s.pt")
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pk1"
                                                     )

    print("=====TRACKING HOOP(S)=====")
    hoop_tracker = HoopTracker("models/best_im.pt")
    hoop_tracks = hoop_tracker.get_object_tracks(video_frames,
                                                read_from_stub=True, stub_path="stubs/hoop_track_stubs.pk1"
                                                )


    print("=====TRACKING BALL=====")
    ball_tracker = BallTracker("models/best_im.pt")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/ball_track_stubs.pk1"
                                                     )
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    print("=====BALL AQUISITION=====")
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    print("=====TRACKING SCORES=====")
    score_tracker = ScoreTracker()
    score_tracks = score_tracker.get_scores(ball_tracks, hoop_tracks)

    print("=====ASSERTING EQUAL LENGTHS=====")
    print(len(video_frames) == len(hoop_tracks) == len(ball_tracks) == len(score_tracks) == len(ball_aquisition))

    print("=====DRAWING VIDEO=====")
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    hoop_tracks_drawer = HoopTracksDrawer()
    score_tracks_drawer = ScoreTracksDrawer()
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    # output_video_frames = hoop_tracks_drawer.draw(output_video_frames, hoop_tracks)
    # output_video_frames = score_tracks_drawer.draw(output_video_frames, score_tracks)

    print("=====SAVING VIDEO=====")
    save_video(output_video_frames, "output_videos/im_score_tracks.mp4")

if __name__ == "__main__":
    main()