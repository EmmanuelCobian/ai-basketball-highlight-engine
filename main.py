from collections import deque
from utils import stream_video_frames, get_video_info, StreamingVideoWriter
from trackers import PlayerTracker, BallTracker, HoopTracker, StreamingScoreTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from ball_aquisition import BallAcquisitionDetector
import cv2

def main():
    INPUT_VIDEO_PATH = "input_videos/im_1.mov"
    OUTPUT_VIDEO_PATH = "output_videos/im_streaming_output.mp4"
    HIGHLIGHTS_FILE_PATH = "highlights.txt"
    
    print("=====GETTING VIDEO INFO=====")
    video_info = get_video_info(INPUT_VIDEO_PATH)
    print(f"Video info: {video_info}")
    
    player_tracker = PlayerTracker("yolo11s.pt")
    ball_hoop_tracker = BallTracker("models/best_im.pt")
    ball_acquisition_detector = BallAcquisitionDetector()
    
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    
    video_writer = StreamingVideoWriter(
        OUTPUT_VIDEO_PATH,
        video_info['width'],
        video_info['height'],
        video_info['fps']
    )
    
    print("=====PROCESSING FRAMES=====")
    print("Press 'q' to quit, 'p' to pause/resume, 's' to save screenshot")
    frame_count = 0
    paused = False
    try:
        highlights = deque()
        with open(HIGHLIGHTS_FILE_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                start_str, end_str = line.split("-")
                start_str = start_str.strip()
                end_str = end_str.strip()
                def time_to_frame(tstr):
                    h, m, s = map(int, tstr.split(":"))
                    total_seconds = h * 3600 + m * 60 + s
                    return int(total_seconds * video_info['fps'])
                highlights.append((time_to_frame(start_str), time_to_frame(end_str)))
                    
        highlight_possessions = {interval: {} for interval in highlights}
            
        for frame_num, frame in stream_video_frames(INPUT_VIDEO_PATH):
            print(f"Processing frame {frame_num + 1}/{video_info['frame_count']}")
            
            player_track = player_tracker.process_frame(frame)
            ball_track, hoop_track = ball_hoop_tracker.process_frame(frame)
            
            posession_player_id = ball_acquisition_detector.process_frame(player_track, ball_track)

            if highlights:
                interval = highlights[0]
                start_f, end_f = interval
                if start_f <= frame_num <= end_f:
                    if posession_player_id == -1:
                        continue
                    highlight_possessions[interval][posession_player_id] = highlight_possessions[interval].get(posession_player_id, 0) + 1
                while highlights and frame_num > end_f:
                    highlights.popleft()
            
            output_frame = player_drawer.draw_frame(frame, player_track, posession_player_id)
            output_frame = ball_drawer.draw_frame(output_frame, ball_track)
            
            cv2.putText(
                output_frame,
                f"Frame: {frame_num + 1}",
                (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )
            
            cv2.imshow('Basketball Analysis - Real Time', output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested stop with 'q' key")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"Playback {'paused' if paused else 'resumed'}")
            elif key == ord('s'):
                screenshot_path = f"output_videos/screenshot_frame_{frame_num + 1}.png"
                cv2.imwrite(screenshot_path, output_frame)
                print(f"Screenshot saved: {screenshot_path}")
            
            while paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('p'):
                    paused = False
                    print("Playback resumed")
                    break
                elif key == ord('q'):
                    print("User requested stop with 'q' key")
                    break
            
            if key == ord('q'):
                break
            
            video_writer.write_frame(output_frame)
            
            frame_count += 1
    finally:
        video_writer.release()
        cv2.destroyAllWindows()
        
    print(f"=====PROCESSING COMPLETE=====")
    print(f"Processed {frame_count} frames")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
    
    print("=====HIGHLIGHT POSSESSIONS=====")
    for interval, possession_counts in highlight_possessions.items():
        print(f"Interval {interval}:")
        for player_id, count in possession_counts.items():
            print(f"  Player {player_id}: {count} frames of possession")

if __name__ == "__main__":
    main()