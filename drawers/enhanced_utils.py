"""
Enhanced drawing utilities for improved player tracking visualization.
"""

import cv2

def draw_enhanced_tracking_status(frame, tracking_info):
    """
    Draw enhanced tracking status information on the frame.
    
    Args:
        frame: The video frame to draw on
        tracking_info: Dictionary with tracking information
    """
    tracked_id = tracking_info.get('tracked_id')
    original_id = tracking_info.get('original_id')
    confidence = tracking_info.get('confidence', 0.0)
    is_temporary = tracking_info.get('is_temporary', False)
    
    # Position for status text
    y_start = 45
    line_height = 35
    
    # Background rectangle for better readability
    cv2.rectangle(frame, (10, 10), (500, 250), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (500, 250), (255, 255, 255), 2)
    
    # Larger font scale and thickness
    main_font_scale = 1.2
    main_thickness = 3
    sub_font_scale = 1.0
    sub_thickness = 2
    
    # Draw tracking status
    if tracked_id is not None:
        if is_temporary:
            status_text = f"TEMP TRACKING: ID {tracked_id}"
            color = (0, 165, 255)  # Orange
        else:
            status_text = f"TRACKING: ID {tracked_id}"
            color = (0, 255, 0)  # Green
            
        cv2.putText(frame, status_text, (15, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, color, main_thickness)
        
        # Show confidence
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (15, y_start + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (255, 255, 255), sub_thickness)
        
        # Show original ID if different
        if original_id != tracked_id:
            orig_text = f"Original ID: {original_id}"
            cv2.putText(frame, orig_text, (15, y_start + 2 * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (255, 255, 255), sub_thickness)
            
            # Show original lost frames if tracking temporary
            original_lost_frames = tracking_info.get('original_lost_frames', 0)
            max_lost_frames = tracking_info.get('max_lost_frames', 15)
            if is_temporary and original_lost_frames > 0:
                lost_text = f"Original lost: {original_lost_frames}/{max_lost_frames}"
                color = (0, 255, 255) if original_lost_frames <= max_lost_frames else (0, 0, 255)
                cv2.putText(frame, lost_text, (15, y_start + 3 * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, color, sub_thickness)
    else:
        # No tracking
        status_text = "NO PLAYER TRACKED"
        cv2.putText(frame, status_text, (15, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, (0, 0, 255), main_thickness)
