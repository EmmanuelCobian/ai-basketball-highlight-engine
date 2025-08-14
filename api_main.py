import os
import cv2
import uuid
import shutil
import tempfile
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from utils import get_video_info
from utils.enhanced_player_tracker import EnhancedPlayerTracker
from trackers import PlayerTracker, BallTracker
from ball_aquisition import BallAcquisitionDetector
from highlight_engine import generate_highlights_frames


HIGHLIGHTS_FILE_PATH = "highlights.txt"
PLAYER_MODEL_PATH = "yolo11s.pt"
BALL_MODEL_PATH = "models/best_im.pt"


app = FastAPI(title="Basketball Highlight Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict[str, Any]] = {}


def _to_py_list(nums):
    if nums is None:
        return None
    try:
        return [float(x) for x in list(nums)]
    except Exception:
        return list(nums)


def _player_list_from_track(player_track: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": int(pid),
            "bbox": _to_py_list(data.get("bbox")),
            "center": _to_py_list(data.get("bbox_center")),
            "confidence": float(data.get("confidence", 0.0)),
        }
        for pid, data in player_track.items()
    ]


async def _send_status(ws: WebSocket, frame_num: int, frame_total: int, message: str, fps: float) -> None:
    await ws.send_json({
        "type": "status_update",
        "frame_num": int(frame_num),
        "frame_current": int(frame_num),
        "frame_total": int(frame_total),
        "fps": float(fps),
        "message": message,
    })


async def _request_user_input(
    ws: WebSocket,
    input_type: str,
    data: Dict[str, Any],
    frame_num: int,
    fps: float,
) -> Dict[str, Any]:
    await ws.send_json({
        "type": "user_input_required",
        "input_type": input_type,
        "frame_num": int(frame_num),
        "fps": float(fps),
        "data": data,
    })
    while True:
        msg = await ws.receive_json()
        if isinstance(msg, dict) and msg.get("response_type"):
            return msg


async def _process_video_session(session: Dict[str, Any], ws: WebSocket) -> None:
    video_path: str = session["video_path"]

    # 1) Get video info
    try:
        video_info = get_video_info(video_path)
        frame_total = int(video_info.get("frame_count", 0))
        fps = float(video_info.get("fps", 24.0))
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Failed to read video info: {e}", "frame_num": 0, "fps": 0.0})
        await ws.close(code=1011)
        return

    try:
        intervals = await asyncio.to_thread(generate_highlights_frames, video_path)
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Failed to generate highlights: {e}", "frame_num": 0, "fps": float(fps)})
        await ws.close(code=1011)
        return

    intervals = list(sorted(intervals))
    highlight_possessions: Dict[Tuple[int, int], Dict[int, int]] = {tuple(iv): {} for iv in intervals}

    # 2) Initialize trackers (offload heavy loads to thread pool)
    try:
        player_tracker = await asyncio.to_thread(PlayerTracker, PLAYER_MODEL_PATH)
        ball_hoop_tracker = await asyncio.to_thread(BallTracker, BALL_MODEL_PATH)
        ball_acquisition_detector = BallAcquisitionDetector()
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Failed to initialize models: {e}", "frame_num": 0, "fps": float(fps)})
        await ws.close(code=1011)
        return

    enhanced_tracker = EnhancedPlayerTracker(max_lost_frames=int(video_info.get("fps", 24) * 2), confidence_threshold=0.25)
    tracking_initialized = False

    frame_count = 0
    current_tracked_id: Optional[int] = None

    # Initialize frame_num before loop for error reporting
    frame_num = 0

    await _send_status(ws, 0, frame_total, "Starting processing", fps)

    try:
        async def wait_for_selection_initial(cur_frame_num: int, player_track: Dict[int, Dict[str, Any]]) -> Optional[int]:
            available_players = _player_list_from_track(player_track)
            if not available_players:
                return None
            resp = await _request_user_input(ws, "player_selection", {
                "available_players": available_players,
                "message": "Select initial player to track",
            }, cur_frame_num, fps)
            # Expect { response_type: "player_selection", player_id: int }
            pid = resp.get("player_id")
            return int(pid) if pid is not None else None

        async def wait_for_temp_confirmation(cur_frame_num: int, original_id: int, current_id: int, player_track: Dict[int, Dict[str, Any]]) -> bool:
            data = {
                "original_id": int(original_id),
                "current_id": int(current_id),
                "original_bbox": _to_py_list(player_track.get(original_id, {}).get("bbox")),
                "current_bbox": _to_py_list(player_track.get(current_id, {}).get("bbox")),
                "message": f"Confirm temporary assignment: keep tracking {int(current_id)} as permanent replacement for {int(original_id)}?",
            }
            resp = await _request_user_input(ws, "confirmation", data, cur_frame_num, fps)
            # Expect { response_type: "confirmation", confirmed: bool }
            return bool(resp.get("confirmed", False))

        async def wait_for_reassignment(cur_frame_num: int, player_track: Dict[int, Dict[str, Any]], suggestions: List[Tuple[int, float]]) -> Optional[int]:
            # suggestions: list of (id, confidence)
            available_players = _player_list_from_track(player_track)
            sugg_struct = [
                {
                    "id": int(pid),
                    "confidence": float(conf),
                    "bbox": _to_py_list(player_track.get(pid, {}).get("bbox")),
                }
                for pid, conf in suggestions
            ]
            current_ctx = None
            if enhanced_tracker.tracking_state and enhanced_tracker.tracking_state.current_id in player_track:
                cid = enhanced_tracker.tracking_state.current_id
                current_ctx = {"id": int(cid), "bbox": _to_py_list(player_track.get(cid, {}).get("bbox"))}
            response = await _request_user_input(ws, "reassignment_selection", {
                "available_players": available_players,
                "suggestions": sugg_struct,
                "current_tracked": current_ctx,
                "message": "Choose a player to reassign or continue without tracking (choice 0)",
            }, cur_frame_num, fps)
            if response.get("choice") == 0 or response.get("continue"):
                return None
            if "player_id" in response and response.get("player_id") is not None:
                return int(response.get("player_id"))
            if "suggestion_index" in response and response.get("suggestion_index") is not None:
                idx = int(response["suggestion_index"]) - 1
                if 0 <= idx < len(suggestions):
                    return int(suggestions[idx][0])
            return None

        cap = cv2.VideoCapture(video_path)
        try:
            while True:
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    break

                # 3) Process frame (offload heavy inference)
                player_track = await asyncio.to_thread(player_tracker.process_frame, frame)
                ball_track, hoop_track = await asyncio.to_thread(ball_hoop_tracker.process_frame, frame)
                posession_player_id = ball_acquisition_detector.process_frame(player_track, ball_track)
                cur_player_ids = set(player_track.keys())

                # 4) Initial selection flow
                if not tracking_initialized:
                    if cur_player_ids:
                        selected = await wait_for_selection_initial(frame_num, player_track)
                        if selected is not None and selected in cur_player_ids:
                            enhanced_tracker.initialize_tracking(selected, player_track[selected]['bbox_center'])
                            tracking_initialized = True
                            current_tracked_id = selected
                            await _send_status(ws, frame_num, frame_total, f"Tracking player ID: {int(selected)}", fps)
                        else:
                            await _send_status(ws, frame_num, frame_total, "Waiting for valid initial selection", fps)
                            frame_count += 1
                            frame_num += 1
                            await asyncio.sleep(0)
                            continue
                    else:
                        await _send_status(ws, frame_num, frame_total, "No players detected in frame. Waiting...", fps)
                        frame_count += 1
                        frame_num += 1
                        await asyncio.sleep(0)
                        continue

                # 5) Update enhanced tracking and handle user input events
                current_tracked_id, status_message, needs_user_input = enhanced_tracker.update_tracking(player_track)
                if needs_user_input:
                    if enhanced_tracker.tracking_state and enhanced_tracker.tracking_state.is_temporary_assignment:
                        original_id = enhanced_tracker.tracking_state.original_id
                        temp_id = enhanced_tracker.tracking_state.current_id
                        confirmed = await wait_for_temp_confirmation(frame_num, original_id, temp_id, player_track)
                        if confirmed:
                            enhanced_tracker.confirm_temporary_as_permanent()
                            current_tracked_id = enhanced_tracker.tracking_state.current_id
                            await _send_status(ws, frame_num, frame_total, f"Temporary assignment confirmed. Tracking {int(current_tracked_id)}", fps)
                        else:
                            enhanced_tracker.deny_temporary_assignment()

                    # Regular reassignment selection
                    if (not enhanced_tracker.tracking_state.is_temporary_assignment or 
                        enhanced_tracker.tracking_state.lost_frames > enhanced_tracker.max_lost_frames):
                        suggestions = enhanced_tracker.get_reassignment_suggestions(player_track, top_n=3)
                        chosen_id = await wait_for_reassignment(frame_num, player_track, suggestions)
                        if chosen_id is None:
                            current_tracked_id = None
                            await _send_status(ws, frame_num, frame_total, "Continuing without player-specific tracking", fps)
                        elif chosen_id in cur_player_ids:
                            enhanced_tracker.confirm_reassignment(chosen_id, player_track)
                            current_tracked_id = chosen_id
                            await _send_status(ws, frame_num, frame_total, f"Reassigned to player {int(chosen_id)}", fps)
                        else:
                            await _send_status(ws, frame_num, frame_total, f"Invalid reassignment choice: {chosen_id}", fps)

                # 6) Highlight possession tallying during intervals
                if intervals:
                    current = intervals[0]
                    start_f, end_f = current
                    if start_f <= frame_num <= end_f:
                        if posession_player_id != -1:
                            highlight_possessions[tuple(current)][int(posession_player_id)] = highlight_possessions[tuple(current)].get(int(posession_player_id), 0) + 1
                    while intervals and frame_num > end_f:
                        intervals.pop(0)

                # 7) Periodic status update
                if frame_num % 10 == 0 or frame_num == frame_total - 1:
                    await _send_status(ws, frame_num, frame_total, "Processing frames...", fps)

                frame_count += 1
                frame_num += 1
                await asyncio.sleep(0)
        finally:
            cap.release()

        # 8) Build final summary
        tracked_player_highlights = 0
        total_highlights = 0
        id_history = [int(x) for x in (enhanced_tracker.tracking_state.id_history if enhanced_tracker.tracking_state else [])]
        highlight_summaries: List[Dict[str, Any]] = []

        for interval, possession_counts in highlight_possessions.items():
            start_frame, end_frame = interval
            if not possession_counts:
                highlight_summaries.append({
                    "interval": [int(start_frame), int(end_frame)],
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
                "possessions": {str(int(pid)): int(cnt) for pid, cnt in possession_counts.items()},
                "winner": {"player_id": int(winner_player_id), "frames": int(winner_frames)},
                "tracked_player_won": bool(tracked_won),
            })

        summary = {
            "processed_frames": int(frame_count),
            "tracked_player_ids": id_history or None,
            "tracked_player_highlights": int(tracked_player_highlights),
            "total_highlights": int(total_highlights),
            "highlights": highlight_summaries,
        }
        await ws.send_json({"type": "completed", "frame_num": int(frame_num), "fps": float(fps), "summary": summary})
        await ws.close(code=1000)

    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Processing failed: {e}", "frame_num": int(frame_num), "fps": float(fps)})
        await ws.close(code=1011)
    finally:
        try:
            temp_dir = session.get("temp_dir")
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@app.post("/sessions")
async def create_session(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp(prefix="bb_session_")
    session_id = str(uuid.uuid4())

    try:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to save uploaded file: {e}")

    SESSIONS[session_id] = {
        "video_path": file_path,
        "temp_dir": temp_dir,
        "status": "uploaded",
    }

    return JSONResponse({"session_id": session_id})


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    session = SESSIONS.get(session_id, {})
    if not session:
        await ws.send_json({"type": "error", "message": "Invalid session_id", "frame_num": 0, "fps": 0.0})
        await ws.close(code=1008)
        return

    session["status"] = "processing"
    await _process_video_session(session, ws)

    SESSIONS.pop(session_id, None)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Welcome to the goplai API!"}


if __name__ == "__main__":
    # Optional: run with `python api_main.py` for quick local testing
    import uvicorn
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=False)
