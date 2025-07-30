from utils import stream_video_frames, get_video_info, read_highlights, find_closest_player, StreamingVideoWriter
from trackers import PlayerTracker, BallTracker, HoopTracker, StreamingScoreTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from ball_aquisition import BallAcquisitionDetector
import cv2
from drawers.utils import draw_frame_num, draw_highlight_detection
import math

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
    
    print("\n\n\n=====PROCESSING FRAMES=====")
    print("Press 'q' to quit, 'p' to pause/resume, 's' to save screenshot")
    frame_count = 0
    paused = False
    try:
        highlights = read_highlights(HIGHLIGHTS_FILE_PATH, video_info['fps'])
        highlight_possessions = {interval: {} for interval in highlights}
        tracking_id = -1
        original_tracking_id = -1
        tracking_lost_frames = 0
        max_lost_frames = 30
        last_known_position = None
            
        for frame_num, frame in stream_video_frames(INPUT_VIDEO_PATH):
            player_track = player_tracker.process_frame(frame)
            ball_track, hoop_track = ball_hoop_tracker.process_frame(frame)
            posession_player_id = ball_acquisition_detector.process_frame(player_track, ball_track)
            
            cur_player_ids = set(player_track.keys())
            
            if tracking_id == -1:
                print(f"Initial frame - Current player IDs: {sorted(cur_player_ids)}")
                try:
                    user_input = input("Enter the player ID to track for highlights: ")
                    tracking_id_candidate = int(user_input)
                    if tracking_id_candidate in cur_player_ids:
                        tracking_id = tracking_id_candidate
                        original_tracking_id = tracking_id_candidate
                        print(f"Tracking player ID: {tracking_id}")
                        if tracking_id in player_track:
                            last_known_position = player_track[tracking_id]['bbox_center']
                    else:
                        print(f"Player ID {tracking_id_candidate} not found in current frame. Available IDs: {sorted(cur_player_ids)}")
                except ValueError:
                    print("Invalid input. Please enter a valid integer player ID.")
            else:
                # PRIORITY 1: Always check if the original player has returned first
                if original_tracking_id in cur_player_ids and tracking_id != original_tracking_id:
                    print(f"Original player {original_tracking_id} has returned! Switching back from {tracking_id}")
                    tracking_id = original_tracking_id
                    tracking_lost_frames = 0
                    last_known_position = player_track[tracking_id]['bbox_center']
                
                # PRIORITY 2: Check if current tracked player (original or substitute) is present
                elif tracking_id in cur_player_ids:
                    # Player is still being tracked - reset lost frames counter
                    tracking_lost_frames = 0
                    last_known_position = player_track[tracking_id]['bbox_center']
                
                # PRIORITY 3: Current player is lost - try recovery
                else:
                    # Player is lost - increment counter
                    tracking_lost_frames += 1
                    print(f"Tracked player {tracking_id} lost for {tracking_lost_frames} frames")
                    
                    # Try to find closest player (but only if we're still tracking the original)
                    if (last_known_position is not None and 
                        tracking_lost_frames <= max_lost_frames and 
                        tracking_id == original_tracking_id):
                        closest_player_id = find_closest_player(player_track, last_known_position)
                        if closest_player_id is not None:
                            print(f"Attempting to reassign tracking from {tracking_id} to {closest_player_id} (temporary)")
                            tracking_id = closest_player_id
                            tracking_lost_frames = 0
                            last_known_position = player_track[tracking_id]['bbox_center']

                    # Check for permanent loss
                    if tracking_lost_frames > max_lost_frames:
                        if tracking_id == original_tracking_id:
                            print(f"Original player {original_tracking_id} has been permanently lost. No substitute found.")
                        else:
                            print(f"Substitute player {tracking_id} has been lost. Continuing without player-specific tracking.")
                        print(f"Player {tracking_id} has been lost for too long. Continuing without player-specific tracking.")
            
            if highlights:
                interval = highlights[0]
                start_f, end_f = interval
                if start_f <= frame_num <= end_f:
                    draw_highlight_detection(frame, 3, 6, (0, 255, 0))
                    if posession_player_id == -1:
                        continue
                    highlight_possessions[interval][posession_player_id] = highlight_possessions[interval].get(posession_player_id, 0) + 1
                while highlights and frame_num > end_f:
                    highlights.popleft()
            
            if tracking_id != -1:
                tracking_status = "Active"
                status_color = (0, 255, 0)  # Green
                
                if 0 < tracking_lost_frames <= max_lost_frames:
                    tracking_status = f"Lost ({tracking_lost_frames}f)"
                    status_color = (0, 165, 255)  # Orange
                elif tracking_lost_frames > max_lost_frames:
                    tracking_status = "Permanently Lost"
                    status_color = (0, 0, 255)  # Red
                
                player_indicator = f"Player: {tracking_id}"
                if tracking_id != original_tracking_id:
                    player_indicator = f"Player: {tracking_id} (substitute for {original_tracking_id})"
                    status_color = (255, 255, 0)  # Cyan for substitute tracking
                
                cv2.putText(
                    frame,
                    f"Tracking {player_indicator} ({tracking_status})",
                    (50, frame.shape[0] - 50),  # Bottom left corner
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    status_color,
                    2,
                    cv2.LINE_AA
                )
                
            draw_frame_num(frame, frame_num, 3, 6, (0, 0, 0))
            output_frame = player_drawer.draw_frame(frame, player_track, posession_player_id)
            output_frame = ball_drawer.draw_frame(output_frame, ball_track)
            
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
        
    print(f"\n\n\n=====PROCESSING COMPLETE=====")
    print(f"Processed {frame_count} frames")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
    
    print("\n\n\n=====HIGHLIGHT POSSESSIONS=====")
    tracked_player_highlights = 0
    total_highlights = 0
    
    for interval, possession_counts in highlight_possessions.items():
        start_frame, end_frame = interval
        print(f"Interval {interval} (frames {start_frame}-{end_frame}):")
        
        if not possession_counts:
            print("  No possession detected in this highlight")
            continue
            
        max_possession_player = max(possession_counts.items(), key=lambda x: x[1])
        winner_player_id, winner_frames = max_possession_player
        for player_id, count in possession_counts.items():
            print(f"  Player {player_id}: {count} frames of possession")
        
        print(f"  Winner: Player {winner_player_id} with {winner_frames} frames")
        if winner_player_id == original_tracking_id:
            tracked_player_highlights += 1
            print(f"  ✓ Tracked player {original_tracking_id} won this highlight!")
        else:
            print(f"  ✗ Tracked player {original_tracking_id} did not win this highlight")
            
        total_highlights += 1
        print()
    
    print("\n\n\n=====TRACKING SUMMARY=====")
    print(f"Tracked Player ID: {original_tracking_id}")
    print(f"Highlights won by tracked player: {tracked_player_highlights}")
    print(f"Total highlights: {total_highlights}")

if __name__ == "__main__":
    main()