"""
WebSocket service for real-time communication during video processing.

This module handles all WebSocket communication including:
- Status updates
- User input requests  
- Error notifications
- Connection management
"""
import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from fastapi import WebSocket, WebSocketDisconnect

from api.models import (
    PlayerTrack, PlayerSuggestion, MessageType, InputType,
    WebSocketStatusUpdate, WebSocketUserInput, WebSocketError, 
    WebSocketCompletion, PlayerInfo
)
from api.config import USER_INPUT_TIMEOUT, HEARTBEAT_INTERVAL


class WebSocketService:
    """Service for handling WebSocket communication."""
    
    def __init__(self):
        """Initialize WebSocket service."""
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """
        Accept WebSocket connection and store it.
        
        Args:
            websocket: WebSocket connection
            session_id: Session identifier
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        """
        Remove WebSocket connection.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_status_update(
        self, 
        websocket: WebSocket, 
        frame_num: int, 
        frame_total: int, 
        message: str, 
        fps: float
    ) -> None:
        """
        Send processing status update to client.
        
        Args:
            websocket: WebSocket connection
            frame_num: Current frame number
            frame_total: Total frames in video
            message: Status message
            fps: Processing frames per second
        """
        try:
            status_update = WebSocketStatusUpdate(
                frame_num=frame_num,
                frame_total=frame_total,
                fps=fps,
                message=message
            )
            await websocket.send_text(status_update.json())
        except Exception as e:
            print(f"Failed to send status update: {e}")
            raise
    
    async def send_heartbeat(self, websocket: WebSocket) -> None:
        """
        Send heartbeat to keep WebSocket connection alive.
        
        Args:
            websocket: WebSocket connection
        """
        try:
            heartbeat = {"type": MessageType.HEARTBEAT, "timestamp": asyncio.get_event_loop().time()}
            await websocket.send_text(json.dumps(heartbeat))
        except Exception as e:
            print(f"Failed to send heartbeat: {e}")
            raise
    
    async def send_error(
        self, 
        websocket: WebSocket, 
        message: str, 
        frame_num: int = 0, 
        fps: float = 0.0
    ) -> None:
        """
        Send error message to client.
        
        Args:
            websocket: WebSocket connection
            message: Error message
            frame_num: Frame number where error occurred
            fps: Processing FPS at time of error
        """
        try:
            error = WebSocketError(
                message=message,
                frame_num=frame_num,
                fps=fps
            )
            await websocket.send_text(error.json())
        except Exception as e:
            print(f"Failed to send error message: {e}")
            raise
    
    async def send_completion(
        self, 
        websocket: WebSocket, 
        frame_num: int, 
        fps: float, 
        summary: Dict[str, Any]
    ) -> None:
        """
        Send processing completion message to client.
        
        Args:
            websocket: WebSocket connection
            frame_num: Final frame number
            fps: Final processing FPS
            summary: Processing summary and results
        """
        try:
            completion = WebSocketCompletion(
                frame_num=frame_num,
                fps=fps,
                summary=summary
            )
            await websocket.send_text(completion.json())
        except Exception as e:
            print(f"Failed to send completion message: {e}")
            raise
    
    async def request_user_input(
        self,
        websocket: WebSocket,
        input_type: str,
        frame_num: int,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Request user input and wait for response.
        
        Args:
            websocket: WebSocket connection
            input_type: Type of input needed
            frame_num: Current frame number
            data: Input-specific data
            
        Returns:
            User's response data
            
        Raises:
            WebSocketDisconnect: If client disconnects
            asyncio.TimeoutError: If user doesn't respond in time
        """
        user_input = WebSocketUserInput(
            input_type=input_type,
            frame_num=frame_num,
            data=data
        )
        await websocket.send_text(user_input.json())
        
        heartbeat_task = asyncio.create_task(self._send_periodic_heartbeat(websocket))
        try:
            response = await asyncio.wait_for(
                websocket.receive_text(), 
                timeout=USER_INPUT_TIMEOUT
            )
            try:
                response_data = json.loads(response)
                return response_data
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON response: {response}")
                
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _send_periodic_heartbeat(self, websocket: WebSocket) -> None:
        """
        Send periodic heartbeats during long operations.
        
        Args:
            websocket: WebSocket connection
        """
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await self.send_heartbeat(websocket)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat failed: {e}")
                break
    
    def _convert_player_track_to_info_list(
        self, 
        player_track: PlayerTrack
    ) -> List[Dict[str, Any]]:
        """
        Convert internal player track format to API response format.
        
        Args:
            player_track: Internal player tracking data
            
        Returns:
            List of player info dictionaries
        """
        return [
            {
                "id": int(pid),
                "bbox": self._to_py_list(data.get("bbox")),
                "center": self._to_py_list(data.get("bbox_center")),
                "confidence": float(data.get("confidence", 0.0)),
            }
            for pid, data in player_track.items()
        ]
    
    def _to_py_list(self, nums) -> Optional[List[float]]:
        """
        Convert various number formats to Python list.
        
        Args:
            nums: Numbers in various formats (numpy, tensor, etc.)
            
        Returns:
            List of floats or None
        """
        if nums is None:
            return None
        try:
            return [float(x) for x in list(nums)]
        except Exception:
            return list(nums)


# Global WebSocket service instance
websocket_service = WebSocketService()
