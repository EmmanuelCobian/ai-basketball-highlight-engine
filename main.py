from utils import stream_video_frames, get_video_info, read_highlights, find_closest_player, StreamingVideoWriter
from utils.enhanced_player_tracker import EnhancedPlayerTracker
from drawers.utils import draw_frame_num, draw_highlight_detection, draw_tracking_status
from drawers.enhanced_utils import draw_enhanced_tracking_status
from trackers import PlayerTracker, BallTracker, HoopTracker, StreamingScoreTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from ball_aquisition import BallAcquisitionDetector
import cv2

def main():
    # INPUT_VIDEO_PATH = "input_videos/im_1.mov"
    INPUT_VIDEO_PATH = "/Users/eman/Desktop/goplai/basketball_analysis-backup/input_videos/video1.mov"
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
    
    # Initialize enhanced tracking
    enhanced_tracker = EnhancedPlayerTracker(max_lost_frames=15, confidence_threshold=0.25)
    tracking_initialized = False
    
    frame_count = 0
    paused = False
    try:
        highlights = read_highlights(HIGHLIGHTS_FILE_PATH, video_info['fps'])
        highlight_possessions = {interval: {} for interval in highlights}
            
        for frame_num, frame in stream_video_frames(INPUT_VIDEO_PATH):
            player_track = player_tracker.process_frame(frame)
            ball_track, hoop_track = ball_hoop_tracker.process_frame(frame)
            posession_player_id = ball_acquisition_detector.process_frame(player_track, ball_track)
            cur_player_ids = set(player_track.keys())
            
            # Handle initial player selection
            if not tracking_initialized:
                if cur_player_ids:
                    print(f"Initial frame - Current player IDs: {sorted(cur_player_ids)}")
                    while True:
                        try:
                            user_input = input("Enter the player ID to track for highlights: ")
                            tracking_id_candidate = int(user_input)
                            if tracking_id_candidate in cur_player_ids:
                                enhanced_tracker.initialize_tracking(
                                    tracking_id_candidate, 
                                    player_track[tracking_id_candidate]['bbox_center']
                                )
                                tracking_initialized = True
                                print(f"Tracking player ID: {tracking_id_candidate}")
                                break
                            else:
                                print(f"Player ID {tracking_id_candidate} not found. Available IDs: {sorted(cur_player_ids)}")
                        except ValueError:
                            print("Invalid input. Please enter a valid integer player ID.")
                else:
                    print("No players detected in frame. Waiting...")
                    continue
            
            current_tracked_id, status_message, needs_user_input = enhanced_tracker.update_tracking(player_track)
            if needs_user_input:
                print(f"\n{status_message}")
                suggestions = enhanced_tracker.get_reassignment_suggestions(player_track, top_n=3)
                
                if suggestions:
                    print("Suggested reassignments (ID: confidence):")
                    for i, (pid, confidence) in enumerate(suggestions, 1):
                        print(f"  {i}. Player {pid}: {confidence:.2f}")
                    print("  0. Continue without player-specific tracking")
                    
                    while True:
                        try:
                            choice = input("Choose an option (0-3) or enter a specific player ID: ")
                            
                            if choice == "0":
                                current_tracked_id = None
                                print("Continuing without player-specific tracking")
                                break
                            elif choice in ["1", "2", "3"]:
                                idx = int(choice) - 1
                                if idx < len(suggestions):
                                    chosen_id = suggestions[idx][0]
                                    enhanced_tracker.confirm_reassignment(chosen_id, player_track)
                                    current_tracked_id = chosen_id
                                    print(f"Reassigned to player {chosen_id}")
                                    break
                            else:
                                chosen_id = int(choice)
                                if chosen_id in cur_player_ids:
                                    enhanced_tracker.confirm_reassignment(chosen_id, player_track)
                                    current_tracked_id = chosen_id
                                    print(f"Reassigned to player {chosen_id}")
                                    break
                                else:
                                    print(f"Player {chosen_id} not found. Available: {sorted(cur_player_ids)}")
                        except ValueError:
                            print("Invalid input. Please try again.")
            
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
            
            tracking_status_info = {
                'tracked_id': current_tracked_id,
                'original_id': enhanced_tracker.tracking_state.original_id if enhanced_tracker.tracking_state else None,
                'confidence': enhanced_tracker.tracking_state.confidence if enhanced_tracker.tracking_state else 0.0,
                'is_temporary': enhanced_tracker.tracking_state.is_temporary_assignment if enhanced_tracker.tracking_state else False
            }
            
            draw_enhanced_tracking_status(frame, tracking_status_info)
            draw_frame_num(frame, frame_num, 3, 6, (0, 255, 0))
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
    
    # Get the original tracking ID for summary
    original_tracked_id = enhanced_tracker.tracking_state.original_id if enhanced_tracker.tracking_state else None
    
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
        if original_tracked_id and winner_player_id == original_tracked_id:
            tracked_player_highlights += 1
            print(f"  ✓ Tracked player {original_tracked_id} won this highlight!")
        else:
            print(f"  ✗ Tracked player {original_tracked_id or 'None'} did not win this highlight")
            
        total_highlights += 1
        print()
    
    print("\n\n\n=====TRACKING SUMMARY=====")
    print(f"Tracked Player ID: {original_tracked_id or 'None'}")
    print(f"Highlights won by tracked player: {tracked_player_highlights}")
    print(f"Total highlights: {total_highlights}")

if __name__ == "__main__":
    main()