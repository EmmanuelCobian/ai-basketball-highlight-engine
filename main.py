from utils import read_video, save_video, stream_video_frames, get_video_info, StreamingVideoWriter
from trackers import PlayerTracker, BallTracker, HoopTracker, ScoreTracker
from trackers import StreamingPlayerTracker, StreamingBallTracker, StreamingHoopTracker, StreamingScoreTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, HoopTracksDrawer, ScoreTracksDrawer
from drawers import StreamingPlayerTracksDrawer, StreamingBallTracksDrawer
from ball_aquisition import BallAquisitionDetector, StreamingBallAcquisitionDetector

def main_streaming():
    """
    Streaming version that processes video frames one at a time.
    """
    # input_video_path = "/Users/eman/Downloads/Untitled.mov"
    input_video_path = "input_videos/im_1.mov"
    output_video_path = "output_videos/im_streaming_output.mp4"
    
    print("=====GETTING VIDEO INFO=====")
    video_info = get_video_info(input_video_path)
    print(f"Video info: {video_info}")
    
    player_tracker = StreamingPlayerTracker("yolo11s.pt")
    ball_hoop_tracker = StreamingBallTracker("models/best_im.pt")
    score_tracker = StreamingScoreTracker()
    ball_acquisition_detector = StreamingBallAcquisitionDetector()
    
    player_drawer = StreamingPlayerTracksDrawer()
    ball_drawer = StreamingBallTracksDrawer()
    
    video_writer = StreamingVideoWriter(
        output_video_path,
        video_info['width'],
        video_info['height'],
        video_info['fps']
    )
    
    print("=====PROCESSING FRAMES=====")
    frame_count = 0
    try:
        for frame_num, frame in stream_video_frames(input_video_path):
            print(f"Processing frame {frame_num + 1}/{video_info['frame_count']}")
            
            player_track = player_tracker.process_frame(frame)
            ball_track, hoop_track = ball_hoop_tracker.process_frame(frame)
            
            ball_acquisition = ball_acquisition_detector.process_frame(player_track, ball_track)
            
            # score = score_tracker.process_frame(ball_track, hoop_track)
            
            output_frame = player_drawer.draw_frame(frame, player_track, ball_acquisition)
            output_frame = ball_drawer.draw_frame(output_frame, ball_track)
            
            video_writer.write_frame(output_frame)
            
            frame_count += 1
    finally:
        video_writer.release()
    
    print(f"=====PROCESSING COMPLETE=====")
    print(f"Processed {frame_count} frames")
    print(f"Output video saved to: {output_video_path}")

def main():
    """
    Original non-streaming version - keeping for compatibility.
    """
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
    # Choose which version to run
    print("Choose processing mode:")
    print("1. Streaming (memory efficient)")
    print("2. Original (load all frames)")
    
    # choice = input("Enter choice (1 or 2): ").strip()
    choice = "1"
    
    if choice == "1":
        main_streaming()
    else:
        main()