"""
Video processing service for basketball analysis.

This module contains the core video processing pipeline including:
- Player and ball tracking
- User interaction handling
- Highlight generation
"""
import os
import cv2
from typing import Dict, Any, List, Optional, Tuple
from fastapi import WebSocket, WebSocketDisconnect

from api.models import SessionData, InputType
from api.services.websocket_service import websocket_service
from api.services.s3_service import s3_service
from api.config import PLAYER_MODEL_PATH, BALL_MODEL_PATH

from utils import get_video_info
from utils.enhanced_player_tracker import EnhancedPlayerTracker
from trackers import PlayerTracker, BallTracker
from ball_aquisition import BallAcquisitionDetector
from highlight_engine import generate_highlights_frames


class VideoProcessingService:
    """Service for processing basketball videos."""
    
    def __init__(self):
        """Initialize video processing service."""
        pass
    
    async def process_video_session(self, session: Dict[str, Any], websocket: WebSocket) -> None:
        """
        Main video processing pipeline.
        
        This method orchestrates the entire video analysis process:
        1. Download video from S3
        2. Initialize trackers and detectors
        3. Process video frame by frame
        4. Handle user interactions
        5. Generate highlights
        6. Clean up resources
        
        Args:
            session: Session data containing S3 key and metadata
            websocket: WebSocket connection for real-time updates
        """
        temp_video_path = None
        
        try:
            # Step 1: Download video from S3
            await websocket_service.send_status_update(
                websocket, 0, 0, "Downloading video from S3...", 0.0
            )
            
            temp_video_path = s3_service.download_video_to_temp(session["s3_key"])
            
            # Step 2: Get video information
            video_info = get_video_info(temp_video_path)
            total_frames = video_info.get('frame_count', 0)
            fps = video_info.get('fps', 30.0)
            
            await websocket_service.send_status_update(
                websocket, 0, total_frames, 
                f"Video loaded: {total_frames} frames @ {fps} FPS", fps
            )
            
            # Step 3: Initialize tracking components
            await websocket_service.send_status_update(
                websocket, 0, total_frames, "Initializing AI models...", fps
            )
            
            player_tracker = PlayerTracker(PLAYER_MODEL_PATH)
            ball_tracker = BallTracker(BALL_MODEL_PATH)
            enhanced_tracker = EnhancedPlayerTracker()
            ball_acquisition = BallAcquisitionDetector()
            
            # Step 4: Process video
            await self._process_video_frames(
                temp_video_path, websocket, total_frames, fps,
                player_tracker, ball_tracker, enhanced_tracker, ball_acquisition
            )
            
        except Exception as e:
            await websocket_service.send_error(websocket, str(e))
            raise
            
        finally:
            # Step 5: Comprehensive Cleanup
            await self._cleanup_processing_resources(session, temp_video_path)

    async def _cleanup_processing_resources(
        self, 
        session: Dict[str, Any], 
        temp_video_path: Optional[str] = None
    ) -> None:
        """
        Comprehensive cleanup of all processing resources.
        
        Cleans up both local temporary files and S3 objects to prevent
        storage accumulation on both the local EC2 instance and S3.
        
        Args:
            session: Session data containing cleanup information
            temp_video_path: Path to temporary video file to clean up
        """
        try:
            # 1. Clean up local temporary video file
            if temp_video_path:
                s3_service.cleanup_temp_file(temp_video_path)
            
            # 2. Clean up any temporary directories (if applicable)
            # Note: The current implementation uses individual temp files
            # but this handles the case if temp directories are used elsewhere
            temp_dir = session.get("temp_dir") if isinstance(session, dict) else getattr(session, 'temp_dir', None)
            if temp_dir and os.path.isdir(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 3. Clean up S3 object - this is important for cost management
            # and prevents accumulation of processed videos in S3
            s3_key = session.get("s3_key") if isinstance(session, dict) else getattr(session, 's3_key', None)
            if s3_key:
                print(f"Attempting to delete S3 object: {s3_key}")
                success = s3_service.cleanup_s3_object(s3_key)
                if success:
                    print(f"Successfully deleted S3 object: {s3_key}")
                else:
                    print(f"Failed to delete S3 object: {s3_key}")
            else:
                print("Warning: No s3_key found in session data for cleanup")
                    
        except Exception as cleanup_error:
            # Never let cleanup errors break the main processing flow
            print(f"Warning: Cleanup operation failed: {cleanup_error}")
    
    async def _process_video_frames(
        self,
        video_path: str,
        websocket: WebSocket,
        total_frames: int,
        fps: float,
        player_tracker: PlayerTracker,
        ball_tracker: BallTracker,
        enhanced_tracker: EnhancedPlayerTracker,
        ball_acquisition: BallAcquisitionDetector
    ) -> None:
        """
        Process video frames with tracking and user interaction.
        
        Args:
            video_path: Path to video file
            websocket: WebSocket connection
            total_frames: Total number of frames
            fps: Video FPS
            player_tracker: Player tracking instance
            ball_tracker: Ball tracking instance
            enhanced_tracker: Enhanced player tracking instance
            ball_acquisition: Ball acquisition detection instance
        """
        import time
        import asyncio
        
        # Constants for processing
        MAX_PROCESSING_GAP = 5.0  # seconds
        
        # Initialize tracking variables
        tracking_initialized = False
        current_tracked_id = None
        frame_num = 0
        frame_count = 0
        last_heartbeat = time.time()

        try:
            # Generate highlight intervals
            intervals = generate_highlights_frames(video_path)
            intervals = list(sorted(intervals))
            highlight_possessions: Dict[Tuple[int, int], Dict[int, int]] = {tuple(iv): {} for iv in intervals}

            await websocket_service.send_status_update(websocket, 0, total_frames, "Starting processing", fps)
            
            cap = cv2.VideoCapture(video_path)
            try:
                while True:
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret:
                        break

                    # Process frame with async threading for heavy inference
                    player_track = await asyncio.to_thread(player_tracker.process_frame, frame)
                    ball_track, hoop_track = await asyncio.to_thread(ball_tracker.process_frame, frame)
                    possession_player_id = ball_acquisition.process_frame(player_track, ball_track)
                    cur_player_ids = set(player_track.keys())

                    # Initial player selection workflow
                    if not tracking_initialized:
                        if cur_player_ids:
                            selected = await self._wait_for_initial_selection(websocket, frame_num, player_track)
                            if selected is not None and selected in cur_player_ids:
                                enhanced_tracker.initialize_tracking(selected, player_track[selected]['bbox_center'])
                                tracking_initialized = True
                                current_tracked_id = selected
                                await websocket_service.send_status_update(
                                    websocket, frame_num, total_frames, 
                                    f"Tracking player ID: {int(selected)}", fps
                                )
                            else:
                                await websocket_service.send_status_update(
                                    websocket, frame_num, total_frames, 
                                    "Waiting for valid initial selection", fps
                                )
                                frame_count += 1
                                frame_num += 1
                                await asyncio.sleep(0)
                                continue
                        else:
                            await websocket_service.send_status_update(
                                websocket, frame_num, total_frames, 
                                "No players detected in frame. Waiting...", fps
                            )
                            frame_count += 1
                            frame_num += 1
                            await asyncio.sleep(0)
                            continue

                    # Enhanced tracking with full user interaction workflow
                    current_tracked_id, status_message, needs_user_input = enhanced_tracker.update_tracking(player_track)
                    
                    if needs_user_input:
                        await self._handle_tracking_user_input(
                            websocket, frame_num, total_frames, fps, 
                            enhanced_tracker, player_track, cur_player_ids
                        )
                        # Get updated tracking ID after user input handling
                        if enhanced_tracker.tracking_state:
                            current_tracked_id = enhanced_tracker.tracking_state.current_id

                    # if status_message:
                    #     await websocket_service.send_status_update(
                    #         websocket, frame_num, total_frames, status_message, fps
                    #     )

                    # Highlight possession tallying during intervals
                    if intervals:
                        current_interval = intervals[0]
                        start_f, end_f = current_interval
                        if start_f <= frame_num <= end_f:
                            if possession_player_id != -1:
                                interval_key = tuple(current_interval)
                                highlight_possessions[interval_key][int(possession_player_id)] = (
                                    highlight_possessions[interval_key].get(int(possession_player_id), 0) + 1
                                )
                        # Remove completed intervals
                        while intervals and frame_num > end_f:
                            intervals.pop(0)

                    # Periodic status updates and heartbeat management
                    current_time = time.time()
                    
                    if frame_num % 10 == 0 or frame_num == total_frames - 1:
                        await websocket_service.send_status_update(
                            websocket, frame_num, total_frames, "Processing frames...", fps
                        )
                        last_heartbeat = current_time
                    elif current_time - last_heartbeat > MAX_PROCESSING_GAP:
                        await websocket_service.send_heartbeat(websocket)
                        last_heartbeat = current_time

                    frame_count += 1
                    frame_num += 1
                    await asyncio.sleep(0)  # Yield control
                    
            finally:
                cap.release()

            # Generate comprehensive final summary
            await self._generate_final_summary(
                websocket, frame_num, frame_count, fps, 
                highlight_possessions, enhanced_tracker
            )
            
        except WebSocketDisconnect:
            # Client disconnected - still need to cleanup resources
            print(f"WebSocket disconnected during video processing")
            return
        except Exception as e:
            await websocket_service.send_error(websocket, str(e))
            raise
    
    async def _wait_for_initial_selection(
        self, 
        websocket: WebSocket, 
        frame_num: int, 
        player_track: Dict[int, Dict[str, Any]]
    ) -> Optional[int]:
        """
        Handle initial player selection by user.
        
        Args:
            websocket: WebSocket connection
            frame_num: Current frame number  
            player_track: Available players in current frame
            
        Returns:
            Selected player ID or None
        """
        available_players = self._player_list_from_track(player_track)
        if not available_players:
            return None
            
        response = await websocket_service.request_user_input(
            websocket, InputType.PLAYER_SELECTION, frame_num,
            {
                "available_players": available_players,
                "message": "Select initial player to track"
            }
        )
        
        pid = response.get("player_id")
        return int(pid) if pid is not None else None
    
    async def _handle_tracking_user_input(
        self,
        websocket: WebSocket,
        frame_num: int, 
        total_frames: int,
        fps: float,
        enhanced_tracker: EnhancedPlayerTracker,
        player_track: Dict[int, Dict[str, Any]],
        cur_player_ids: set
    ) -> None:
        """
        Handle all user input needs for enhanced tracking.
        
        Args:
            websocket: WebSocket connection
            frame_num: Current frame number
            total_frames: Total frames for progress
            fps: Processing FPS
            enhanced_tracker: Enhanced player tracker instance
            player_track: Current player tracking data
            cur_player_ids: Set of current player IDs
        """
        # Handle temporary assignment confirmation
        if (enhanced_tracker.tracking_state and 
            enhanced_tracker.tracking_state.is_temporary_assignment):
            
            original_id = enhanced_tracker.tracking_state.original_id
            temp_id = enhanced_tracker.tracking_state.current_id
            
            confirmed = await self._wait_for_temp_confirmation(
                websocket, frame_num, original_id, temp_id, player_track
            )
            
            if confirmed:
                enhanced_tracker.confirm_temporary_as_permanent()
                current_tracked_id = enhanced_tracker.tracking_state.current_id
                await websocket_service.send_status_update(
                    websocket, frame_num, total_frames,
                    f"Temporary assignment confirmed. Tracking {int(current_tracked_id)}", fps
                )
            else:
                enhanced_tracker.deny_temporary_assignment()

        # Handle regular reassignment selection
        if (not enhanced_tracker.tracking_state.is_temporary_assignment or 
            enhanced_tracker.tracking_state.lost_frames > enhanced_tracker.max_lost_frames):
            
            suggestions = enhanced_tracker.get_reassignment_suggestions(player_track, top_n=3)
            chosen_id = await self._wait_for_reassignment(
                websocket, frame_num, player_track, suggestions, enhanced_tracker
            )
            
            if chosen_id is None:
                await websocket_service.send_status_update(
                    websocket, frame_num, total_frames,
                    "Continuing without player-specific tracking", fps
                )
            elif chosen_id in cur_player_ids:
                enhanced_tracker.confirm_reassignment(chosen_id, player_track)
                await websocket_service.send_status_update(
                    websocket, frame_num, total_frames,
                    f"Reassigned to player {int(chosen_id)}", fps
                )
            else:
                await websocket_service.send_status_update(
                    websocket, frame_num, total_frames,
                    f"Invalid reassignment choice: {chosen_id}", fps
                )
    
    async def _wait_for_temp_confirmation(
        self,
        websocket: WebSocket,
        frame_num: int,
        original_id: int,
        current_id: int,
        player_track: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Wait for user confirmation of temporary assignment.
        
        Args:
            websocket: WebSocket connection
            frame_num: Current frame number
            original_id: Original player ID being tracked
            current_id: Current temporary player ID
            player_track: Current player tracking data
            
        Returns:
            True if confirmed, False otherwise
        """
        data = {
            "original_id": int(original_id),
            "current_id": int(current_id),
            "original_bbox": self._to_py_list(player_track.get(original_id, {}).get("bbox")),
            "current_bbox": self._to_py_list(player_track.get(current_id, {}).get("bbox")),
            "message": f"Confirm temporary assignment: keep tracking {int(current_id)} as permanent replacement for {int(original_id)}?",
        }
        
        response = await websocket_service.request_user_input(
            websocket, InputType.CONFIRMATION, frame_num, data
        )
        return bool(response.get("confirmed", False))
    
    async def _wait_for_reassignment(
        self,
        websocket: WebSocket,
        frame_num: int,
        player_track: Dict[int, Dict[str, Any]],
        suggestions: List[Tuple[int, float]],
        enhanced_tracker: EnhancedPlayerTracker
    ) -> Optional[int]:
        """
        Wait for user to select player reassignment.
        
        Args:
            websocket: WebSocket connection
            frame_num: Current frame number
            player_track: Current player tracking data
            suggestions: List of (player_id, confidence) suggestions
            enhanced_tracker: Enhanced player tracker instance
            
        Returns:
            Selected player ID or None to continue without tracking
        """
        available_players = self._player_list_from_track(player_track)
        sugg_struct = [
            {
                "id": int(pid),
                "confidence": float(conf),
                "bbox": self._to_py_list(player_track.get(pid, {}).get("bbox")),
            }
            for pid, conf in suggestions
        ]
        
        current_ctx = None
        if (enhanced_tracker.tracking_state and 
            enhanced_tracker.tracking_state.current_id in player_track):
            cid = enhanced_tracker.tracking_state.current_id
            current_ctx = {
                "id": int(cid), 
                "bbox": self._to_py_list(player_track.get(cid, {}).get("bbox"))
            }
        
        response = await websocket_service.request_user_input(
            websocket, InputType.REASSIGNMENT_SELECTION, frame_num, {
                "available_players": available_players,
                "suggestions": sugg_struct,
                "current_tracked": current_ctx,
                "message": "Choose a player to reassign or continue without tracking (choice 0)",
            }
        )
        
        # Handle different response formats
        if response.get("choice") == 0 or response.get("continue"):
            return None
        if "player_id" in response and response.get("player_id") is not None:
            return int(response.get("player_id"))
        if "suggestion_index" in response and response.get("suggestion_index") is not None:
            idx = int(response["suggestion_index"]) - 1
            if 0 <= idx < len(suggestions):
                return int(suggestions[idx][0])
        return None

    async def _generate_final_summary(
        self,
        websocket: WebSocket,
        final_frame: int,
        frame_count: int,
        fps: float,
        highlight_possessions: Dict[Tuple[int, int], Dict[int, int]],
        enhanced_tracker: EnhancedPlayerTracker
    ) -> None:
        """
        Generate comprehensive final summary and send completion.
        
        Args:
            websocket: WebSocket connection
            final_frame: Final frame number processed
            frame_count: Total frames processed
            fps: Processing FPS
            highlight_possessions: Dictionary mapping highlight intervals to possession counts
            enhanced_tracker: Enhanced player tracker for ID history
        """
        await websocket_service.send_status_update(
            websocket, final_frame, final_frame,
            "Generating highlights...", fps
        )
        
        # Build comprehensive summary
        tracked_player_highlights = 0
        total_highlights = 0
        id_history = [
            int(x) for x in (
                enhanced_tracker.tracking_state.id_history 
                if enhanced_tracker.tracking_state 
                else []
            )
        ]
        highlight_summaries: List[Dict[str, Any]] = []

        for interval, possession_counts in highlight_possessions.items():
            start_frame, end_frame = interval
            
            if not possession_counts:
                highlight_summaries.append({
                    "interval": [int(start_frame), int(end_frame)],
                    "start_time": round(start_frame / fps, 2),
                    "end_time": round(end_frame / fps, 2),
                    "duration": round((end_frame - start_frame) / fps, 2),
                    "possessions": {},
                    "winner": None,
                    "tracked_player_won": False,
                })
                continue

            winner_player_id, winner_frames = max(possession_counts.items(), key=lambda x: x[1])
            winner_player_id = int(winner_player_id)
            winner_frames = int(winner_frames)
            tracked_won = winner_player_id in id_history
            
            if tracked_won:
                tracked_player_highlights += 1
            total_highlights += 1

            highlight_summaries.append({
                "interval": [int(start_frame), int(end_frame)],
                "start_time": round(start_frame / fps, 2),
                "end_time": round(end_frame / fps, 2), 
                "duration": round((end_frame - start_frame) / fps, 2),
                "possessions": {str(int(pid)): int(cnt) for pid, cnt in possession_counts.items()},
                "winner": {"player_id": int(winner_player_id), "frames": int(winner_frames)},
                "tracked_player_won": bool(tracked_won),
            })

        summary = {
            "processed_frames": int(frame_count),
            "total_frames": int(final_frame),
            "processing_fps": float(fps),
            "tracked_player_ids": id_history or None,
            "tracked_player_highlights": int(tracked_player_highlights),
            "total_highlights": int(total_highlights),
            "highlights": highlight_summaries,
        }
        
        await websocket_service.send_completion(websocket, final_frame, fps, summary)

    def _player_list_from_track(self, player_track: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert player track to list format for user selection.
        
        Args:
            player_track: Dictionary of player tracking data
            
        Returns:
            List of player info dictionaries
        """
        return [
            {
                "id": int(pid),
                "bbox": self._to_py_list(pdata.get("bbox", [])),
                "confidence": float(pdata.get("confidence", 0.0)),
            }
            for pid, pdata in player_track.items()
        ]
    
    def _to_py_list(self, data) -> List:
        """
        Convert data to Python list format.
        
        Args:
            data: Data to convert (numpy array, tensor, list, etc.)
            
        Returns:
            Python list representation of data
        """
        if data is None:
            return []
        if hasattr(data, 'tolist'):
            return data.tolist()
        return list(data) if data else []

video_service = VideoProcessingService()
