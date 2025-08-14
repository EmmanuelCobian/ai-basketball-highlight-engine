import asyncio
import json
import os
import sys

import requests
import websockets

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"


def upload_video(path: str) -> str:
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "video/mp4")}
        r = requests.post(f"{API_URL}/sessions", files=files, timeout=300)
        r.raise_for_status()
        return r.json()["session_id"]


async def run_session(session_id: str):
    uri = f"{WS_URL}/ws/{session_id}"
    async with websockets.connect(
        uri,
        max_size=8 * 1024 * 1024,
        ping_interval=30,
        ping_timeout=120,
        close_timeout=10,
        max_queue=32,
    ) as ws:
        print("Connected to WebSocket. Waiting for messages...")
        fps_seen = None
        while True:
            try:
                msg = await ws.recv()
            except websockets.exceptions.ConnectionClosedOK:
                print("WebSocket closed (OK)")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"WebSocket error: {e}")
                break

            try:
                data = json.loads(msg)
            except Exception:
                print("Non-JSON message:", msg[:200])
                continue

            mtype = data.get("type")
            fps = data.get("fps", fps_seen)
            if fps is not None:
                fps_seen = fps

            if mtype == "status_update":
                cur = data.get("frame_num", data.get("frame_current"))
                total = data.get("frame_total")
                print(f"Status: frame {cur}/{total} @ {fps_seen} fps - {data.get('message')}")

            elif mtype == "user_input_required":
                itype = data.get("input_type")
                payload = data.get("data", {})
                fnum = data.get("frame_num")
                print(f"User input required at frame {fnum}: {itype}")

                # Minimal auto-responder for demo purposes
                if itype == "player_selection":
                    avail = payload.get("available_players", [])
                    choice = avail[0]["id"] if avail else None
                    resp = {"response_type": "player_selection", "player_id": choice}
                    await ws.send(json.dumps(resp))

                elif itype == "confirmation":
                    resp = {"response_type": "confirmation", "confirmed": True}
                    await ws.send(json.dumps(resp))

                elif itype == "reassignment_selection":
                    # If suggestions exist, pick the top suggestion
                    sugg = payload.get("suggestions", [])
                    if sugg:
                        resp = {"response_type": "reassignment_selection", "player_id": sugg[0]["id"]}
                    else:
                        resp = {"response_type": "reassignment_selection", "choice": 0}
                    await ws.send(json.dumps(resp))

                else:
                    print("Unknown input type, continuing without selection")
                    await ws.send(json.dumps({"response_type": "reassignment_selection", "choice": 0}))

            elif mtype == "completed":
                print("Processing completed at frame", data.get("frame_num"), "@", fps_seen, "fps. Summary:")
                print(json.dumps(data.get("summary", {}), indent=2))
                break

            elif mtype == "error":
                print("Server error at frame", data.get("frame_num"), "@", fps_seen, "fps:", data.get("message"))
                break

            else:
                print("Unknown message:", data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_example.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print("Uploading video...", video_path)
    session_id = upload_video(video_path)
    print("Session ID:", session_id)

    asyncio.run(run_session(session_id))
